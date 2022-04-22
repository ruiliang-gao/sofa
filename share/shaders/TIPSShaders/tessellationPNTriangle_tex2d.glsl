#version 420 compatibility

struct V2T
{
    vec3 position;
    vec3 normal;
	
	//Texture3D coords
	vec2 texcoord2;
};

struct TC2E
{
    vec3 b3;
    vec3 b21x3, b12x3;
    vec3 n2;
    vec3 n11;
	
	//Texture2D coords
	vec2 bt2;
	//vec3 bt21x3, bt12x3;
};

struct TPatch
{
    //vec3 b300,b030,b003;
    //vec3 b210,b120;
    //vec3 b021,b012;
    //vec3 b102,b201;
    vec3 b111x6;
    //vec3 n200,n020,n002;
    //vec3 n110,n011,n101;
	
	//Texture3D coords - probably not needed, causes glitchy textures
	//vec3 bt111x6;
};

struct T2G
{
    vec3 position;
    vec3 normal;
    vec3 patchDistance;
	
	//Texture2D coords
	vec2 texcoord2;
};

struct G2F
{
    vec3 position;
    vec3 normal;
    vec3 patchDistance;
    vec3 triDistance;
	
	//Texture3D coords
	vec2 texcoord2;
};

#ifdef VertexShader //--------------------------------------
out V2T vdata;

void main()
{
    vdata.position = gl_Vertex.xyz;
    vdata.normal = normalize(gl_Normal.xyz);
	
	//Texture3D coords
	vdata.texcoord2.xy = gl_MultiTexCoord0.xy;
}
#endif
#ifdef TessellationControlShader //-----------------------
layout (vertices = 3) out;
uniform float TessellationLevel;
//const float TessellationLevel = 6.;
in V2T vdata[];
out TC2E tcdata[];
out patch TPatch tpdata;

void main()
{
//    const int I = gl_InvocationID;
#define I gl_InvocationID
    const int J = (I+1)%3;
    vec3 p1 = vdata[I].position;
	
	//Texture2D coords
	vec2 t1 = vdata[I].texcoord2;
	
    tcdata[I].b3 = p1;
	
	//Texture2D coords
	tcdata[I].bt2 = t1;
	
    tcdata[I].n2 = vdata[I].normal;
    vec3 dp = vdata[J].position - vdata[I].position;
	
	//Texture3D coords - probably not needed, causes glitchy textures
	//vec3 tdp = vdata[J].texcoord3 - vdata[I].texcoord3;
	
    vec3 n12 = vdata[I].normal + vdata[J].normal;
    tcdata[I].n11 = normalize( n12*dot(dp,dp) + dp*(2.*dot(dp,n12)) );

    float w12 = dot(dp,vdata[I].normal);
    float w21 = -dot(dp,vdata[J].normal);
	
	//Texture3D coords - probably not needed, causes glitchy textures
	//float tw12 = dot(tdp,vdata[I].normal);
    //float tw21 = -dot(tdp,vdata[J].normal);
	
    tcdata[I].b21x3 = (2.*vdata[I].position + vdata[J].position - w12*vdata[I].normal);
    tcdata[I].b12x3 = (2.*vdata[J].position + vdata[I].position - w21*vdata[J].normal);
	
	//Texture3D coords - probably not needed, causes glitchy textures
	//tcdata[I].bt21x3 = (2.*vdata[I].texcoord3 + vdata[J].texcoord3 - tw12*vdata[I].normal);
    //tcdata[I].bt12x3 = (2.*vdata[J].texcoord3 + vdata[I].texcoord3 - tw21*vdata[J].normal);

    gl_TessLevelOuter[I] = TessellationLevel;

    barrier();
    //float ee = (tcdata[0].b21x3[I] + tcdata[0].b12x3[I] + tcdata[1].b21x3[I] + tcdata[1].b12x3[I] + tcdata[2].b21x3[I] + tcdata[2].b12x3[I]);
    //float vv = (tcdata[0].b3[I] + tcdata[1].b3[I] + tcdata[2].b3[I]);
    //p.b111x6[I] = (ee/2. + vv);
    if (I==0)
    {
        vec3 eex18 = (tcdata[0].b21x3 + tcdata[0].b12x3 + tcdata[1].b21x3 + tcdata[1].b12x3 + tcdata[2].b21x3 + tcdata[2].b12x3);
        vec3 vvx3 = (tcdata[0].b3 + tcdata[1].b3 + tcdata[2].b3);
        tpdata.b111x6 = (eex18/2. - vvx3);
		
		//Texture3D coords - probably not needed, causes glitchy textures
		//vec3 teex18 = (tcdata[0].bt21x3 + tcdata[0].bt12x3 + tcdata[1].bt21x3 + tcdata[1].bt12x3 + tcdata[2].bt21x3 + tcdata[2].bt12x3);
		//vec3 tvvx3 = (tcdata[0].bt3 + tcdata[1].bt3 + tcdata[2].bt3);
		//tpdata.bt111x6 = (teex18/2. - tvvx3);

        gl_TessLevelInner[0] = TessellationLevel;

    }
#undef I
}

#endif
#ifdef TessellationEvaluationShader //-----------------------
layout(triangles, equal_spacing, cw) in;
in TC2E tcdata[];
in patch TPatch tpdata;
out T2G tedata;

void main()
{
    tedata.patchDistance = gl_TessCoord;
    float u = gl_TessCoord.x, v = gl_TessCoord.y, w = gl_TessCoord.z;

    float u2 = u*u, v2 = v*v, w2 = w*w;
    tedata.normal = tcdata[0].n2*(u2) + tcdata[1].n2*(v2) + tcdata[2].n2*(w2) + 
        tcdata[0].n11*(u*v) + tcdata[1].n11*(v*w) + tcdata[2].n11*(w*u);
    vec3 pos = tcdata[0].b3*(u2*u) + tcdata[1].b3*(v2*v) + tcdata[2].b3*(w2*w) +
        tcdata[0].b21x3*(u2*v) + tcdata[0].b12x3*(u*v2) +
        tcdata[1].b21x3*(v2*w) + tcdata[1].b12x3*(v*w2) +
        tcdata[2].b21x3*(w2*u) + tcdata[2].b12x3*(w*u2) +
        tpdata.b111x6*(u*v*w);
		
	//Texture3D coords - probably not needed, causes glitchy textures
	//vec3 tpos = tcdata[0].bt3*(u2*u) + tcdata[1].bt3*(v2*v) + tcdata[2].bt3*(w2*w) +
        //tcdata[0].bt21x3*(u2*v) + tcdata[0].bt12x3*(u*v2) +
        //tcdata[1].bt21x3*(v2*w) + tcdata[1].bt12x3*(v*w2) +
        //tcdata[2].bt21x3*(w2*u) + tcdata[2].bt12x3*(w*u2) +
        //tpdata.bt111x6*(u*v*w);
	vec2 tpos = tcdata[0].bt2 * u + tcdata[1].bt2 * v + tcdata[2].bt2 * w;

    //tedata.normal = tcdata[0].n2*(w) + tcdata[1].n2*(u) + tcdata[2].n2*(v);
    //vec3 pos = tcdata[0].b3*(w) + tcdata[1].b3*(u) + tcdata[2].b3*(v);

    tedata.position = pos;
	
	//Texture3D coords
	tedata.texcoord2 = tpos;
	
    gl_Position = gl_ModelViewProjectionMatrix * vec4(pos, 1);
}

#endif
//#ifdef GeometryShader //------------------------------------

// layout(triangles) in;
// layout(triangle_strip, max_vertices = 3) out;
// in T2G tedata[3];
// out G2F gdata;

// void main()
// {
// /*
    // vec3 A = tePosition[2] - tePosition[0];
    // vec3 B = tePosition[1] - tePosition[0];
    // gNormal = normalize(cross(A, B));
// */
    // gdata.position = tedata[0].position;
    // gdata.normal = tedata[0].normal;
    // gdata.patchDistance = tedata[0].patchDistance;
    // gdata.triDistance = vec3(1, 0, 0);
	
	// //Texture2D coords
	// gdata.texcoord2 = tedata[0].texcoord2;
	
    // gl_Position = gl_in[0].gl_Position;
    // EmitVertex();

    // gdata.position = tedata[1].position;
    // gdata.normal = tedata[1].normal;
    // gdata.patchDistance = tedata[1].patchDistance;
    // gdata.triDistance = vec3(0, 1, 0);
	
	// //Texture2D coords
	// gdata.texcoord2 = tedata[1].texcoord2;
	
    // gl_Position = gl_in[1].gl_Position;
    // EmitVertex();

    // gdata.position = tedata[2].position;
    // gdata.normal = tedata[2].normal;
    // gdata.patchDistance = tedata[2].patchDistance;
    // gdata.triDistance = vec3(0, 0, 1);
	
	// //Texture2D coords
	// gdata.texcoord2 = tedata[2].texcoord2;
	
    // gl_Position = gl_in[2].gl_Position;
    // EmitVertex();

    // EndPrimitive();
// }
//#endif
#ifdef FragmentShader //------------------------------------
in T2G tedata;
uniform sampler2D colorTexture2;
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


void main() {
	vec2 in_texcoord2 = tedata.texcoord2.xy;
	vec4 color = texture2D(colorTexture2, in_texcoord2);

	vec4 pos = gl_ModelViewMatrix * vec4(tedata.position, 1);
	vec3 lightDir = normalize(vec3(0.0) - pos.xyz);//light position in ViewCoord

  float distance = length(pos.xyz)/10; //camera = (0,0,0), and distance should be in 'meters'
  float attenuation = 1.0 / (distance * distance);
  vec3 radiance = vec3(1.0) * attenuation;
  float roughness = 0.1;

  //Generating a bump map from noise
  float noiseVal = fnoise(vec3(in_texcoord2,1)) * 0.5 + 0.5;
  float E = 0.001;

  vec3 px = vec3(in_texcoord2,1);
  px.x += E;
  vec3 py = vec3(in_texcoord2,1);
  py.y += E;
  vec3 pz = vec3(in_texcoord2,1);
  pz.z += E;

  vec3 bump = vec3(fnoise(px)*0.5+0.5, fnoise(py)*0.5+0.5, fnoise(pz)*0.5+0.5);
  vec3 grad = vec3((bump.x-noiseVal)/E, (bump.y-noiseVal)/E, (bump.z-noiseVal)/E);

  vec3 pN = normalize(tedata.normal.xyz - grad);

  vec4 Nr = gl_ModelViewMatrixInverseTranspose * vec4(pN, 1);
  vec3 myN = normalize(Nr.xyz);//This step is critical that the Rasteration step will use interpolattion to get N, which must be normalized.
  //vec3 ReflectedRay = reflect(lightDir, myN );
  //vec3 CamDir = normalize(pos.xyz);//Cam position in ViewCoord is 0,0,0
  
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
  gl_FragColor.a = 0.5;
}
#endif