#version 420 compatibility

struct V2F {
  vec3 position;
  vec3 normal;
  vec2 texcoord;
};

#ifdef VertexShader //--------------------------------------
out V2F vdata;

void main()
{
  gl_Position = ftransform();

  vdata.position = gl_Vertex.xyz;
  vdata.normal = normalize(gl_Normal.xyz);
  vdata.texcoord.xy = gl_MultiTexCoord0.xy;
}
#endif

#ifdef FragmentShader //------------------------------------
in V2F vdata;
uniform sampler2D colorTexture;
const float PI = 3.14159265359;

float basicNoise(vec2 co){
    return 0.5 + 0.5 * fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

//3D Value Noise generator by Morgan McGuire @morgan3d
//https://www.shadertoy.com/view/4dS3Wd
float hash(float n) { return fract(sin(n) * 1e4); }
float hash(vec2 p) { return fract(1e4 * sin(17.0 * p.x + p.y * 0.1) * (0.1 + abs(sin(p.y * 13.0 + p.x)))); }

float noise(vec3 x) {
	const vec3 step = vec3(110, 241, 171);

	vec3 i = floor(x);
	vec3 f = fract(x);

	// For performance, compute the base input to a 1D hash from the integer part of the argument and the
	// incremental change to the 1D based on the 3D -> 1D wrapping
  float n = dot(i, step);

	vec3 u = f * f * (3.0 - 2.0 * f);
	return mix(mix(mix( hash(n + dot(step, vec3(0, 0, 0))), hash(n + dot(step, vec3(1, 0, 0))), u.x),
		   mix( hash(n + dot(step, vec3(0, 1, 0))), hash(n + dot(step, vec3(1, 1, 0))), u.x), u.y),
	       mix(mix( hash(n + dot(step, vec3(0, 0, 1))), hash(n + dot(step, vec3(1, 0, 1))), u.x),
		   mix( hash(n + dot(step, vec3(0, 1, 1))), hash(n + dot(step, vec3(1, 1, 1))), u.x), u.y), u.z);
}

//Fractional Brownian Motion
#define NUM_OCTAVES 3

float fnoise(vec3 x) {
	float v = 0.0;
	float a = 0.5;
	vec3 shift = vec3(100);
	for (int i = 0; i < NUM_OCTAVES; ++i) {
		v += a * noise(x);
		x = x * 2.0 + shift;
		a *= 0.5;
	}
	return v;
}

/////PBR helper functions
float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a      = roughness*roughness;
    float a2     = a*a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;
  
    float num   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
  
    return num / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float num   = NdotV;
    float denom = NdotV * (1.0 - k) + k;
  
    return num / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2  = GeometrySchlickGGX(NdotV, roughness);
    float ggx1  = GeometrySchlickGGX(NdotL, roughness);
  
    return ggx1 * ggx2;
}

void main()
{
  vec2 in_texcoord = vdata.texcoord;
  vec4 color = texture2D(colorTexture, in_texcoord); 

	vec4 pos = gl_ModelViewMatrix * vec4(vdata.position, 1);
	vec3 lightDir = normalize(vec3(0.0) - pos.xyz);//light position in ViewCoord

  float distance = length(pos.xyz)/10; //camera = (0,0,0), and distance should be in 'meters'
  float attenuation = 1.0 / (distance * distance);
  vec3 radiance = vec3(1.0) * attenuation;
  float roughness = 0.1;

  //Generating a bump map from noise
  float noiseVal = fnoise(vec3(in_texcoord,1)) * 0.5 + 0.5;
  float E = 0.001;

  vec3 px = vec3(in_texcoord,1);
  px.x += E;
  vec3 py = vec3(in_texcoord,1);
  py.y += E;
  vec3 pz = vec3(in_texcoord,1);
  pz.z += E;

  vec3 bump = vec3(fnoise(px)*0.5+0.5, fnoise(py)*0.5+0.5, fnoise(pz)*0.5+0.5);
  vec3 grad = vec3((bump.x-noiseVal)/E, (bump.y-noiseVal)/E, (bump.z-noiseVal)/E);

  vec3 pN = normalize(vdata.normal.xyz );

  vec4 Nr = gl_ModelViewMatrixInverseTranspose * vec4(pN, 1);
  vec3 myN = normalize(Nr.xyz);//This step is critical that the Rasteration step will use interpolattion to get N, which must be normalized.
  //vec3 ReflectedRay = reflect(lightDir, myN );
  //vec3 CamDir = normalize(pos.xyz);//Cam position in ViewCoord is 0,0,0
  //As light is attached to camera, Halfway dir = View dir = Light Dir
  
  vec3 V = normalize(vec3(0.0) - pos.xyz); //View dir
  vec3 H = lightDir; //normalize(V + lightDir); //Halfway dir = View dir = Light Dir

  //Fresnel Term = kS
  // F0 set based on experience, 0.04 is a common number for water, plastic, etc.
  vec3 F0 = vec3(0.5);
  
  //float base = 1 - dot(CamDir, h);
  float base = 1 - dot(V, myN);
  float exponential = pow(base, 5.0);
  vec3 fresnel = exponential + F0 * (1.0 - exponential);
  float denominator = max(4 * max(dot(myN, V), 0.0) * max(dot(myN, lightDir), 0.0), 0.001);
  //Cook-Torrance BRDF
  vec3 specular = fresnel * DistributionGGX(myN, H, roughness) * GeometrySmith(myN, V, lightDir, roughness) / denominator;

  //kD = (1 - specular) * (1 - metallic)
  vec3 kD = (vec3(1.0) - fresnel) * (1 - 0.05); 
  vec3 diffuse = kD * color.xyz / PI;

  // scale light by NdotL
  float NdotL = max(dot(myN, lightDir), 0.0);

  // color = ambient + diffuse + specular
  gl_FragColor.xyz = 0.05 * color.xyz + (diffuse + specular) * radiance * NdotL; 
  // HDR tonemapping
  gl_FragColor.xyz = gl_FragColor.xyz / (gl_FragColor.xyz + vec3(1.0)); 
  // Gamma correction
  gl_FragColor.xyz = pow(gl_FragColor.xyz, vec3(1.0/2.2)); 

}
#endif
