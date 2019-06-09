/* A Bison parser, made by GNU Bison 2.5.  */

/* Bison interface for Yacc-like parsers in C
   
      Copyright (C) 1984, 1989-1990, 2000-2011 Free Software Foundation, Inc.
   
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.
   
   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */


/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     NUMBER = 258,
     FLOAT = 259,
     STRING = 260,
     NAME = 261,
     ANCHOR = 262,
     APPEARANCE = 263,
     AUDIOCLIP = 264,
     BACKGROUND = 265,
     BILLBOARD = 266,
     BOX = 267,
     COLLISION = 268,
     COLOR = 269,
     COLOR_INTERP = 270,
     COORDINATE = 271,
     COORDINATE_INTERP = 272,
     CYLINDER_SENSOR = 273,
     NULL_STRING = 274,
     CONE = 275,
     CUBE = 276,
     CYLINDER = 277,
     DIRECTIONALLIGHT = 278,
     FONTSTYLE = 279,
     ERROR = 280,
     EXTRUSION = 281,
     ELEVATION_GRID = 282,
     FOG = 283,
     INLINE = 284,
     MOVIE_TEXTURE = 285,
     NAVIGATION_INFO = 286,
     PIXEL_TEXTURE = 287,
     GROUP = 288,
     INDEXEDFACESET = 289,
     INDEXEDLINESET = 290,
     S_INFO = 291,
     LOD = 292,
     MATERIAL = 293,
     NORMAL = 294,
     POSITION_INTERP = 295,
     PROXIMITY_SENSOR = 296,
     SCALAR_INTERP = 297,
     SCRIPT = 298,
     SHAPE = 299,
     SOUND = 300,
     SPOTLIGHT = 301,
     SPHERE_SENSOR = 302,
     TEXT = 303,
     TEXTURE_COORDINATE = 304,
     TEXTURE_TRANSFORM = 305,
     TIME_SENSOR = 306,
     SWITCH = 307,
     TOUCH_SENSOR = 308,
     VIEWPOINT = 309,
     VISIBILITY_SENSOR = 310,
     WORLD_INFO = 311,
     NORMAL_INTERP = 312,
     ORIENTATION_INTERP = 313,
     POINTLIGHT = 314,
     POINTSET = 315,
     SPHERE = 316,
     PLANE_SENSOR = 317,
     TRANSFORM = 318,
     S_CHILDREN = 319,
     S_PARAMETER = 320,
     S_URL = 321,
     S_MATERIAL = 322,
     S_TEXTURETRANSFORM = 323,
     S_TEXTURE = 324,
     S_LOOP = 325,
     S_STARTTIME = 326,
     S_STOPTIME = 327,
     S_GROUNDANGLE = 328,
     S_GROUNDCOLOR = 329,
     S_SPEED = 330,
     S_AVATAR_SIZE = 331,
     S_BACKURL = 332,
     S_BOTTOMURL = 333,
     S_FRONTURL = 334,
     S_LEFTURL = 335,
     S_RIGHTURL = 336,
     S_TOPURL = 337,
     S_SKYANGLE = 338,
     S_SKYCOLOR = 339,
     S_AXIS_OF_ROTATION = 340,
     S_COLLIDE = 341,
     S_COLLIDETIME = 342,
     S_PROXY = 343,
     S_SIDE = 344,
     S_AUTO_OFFSET = 345,
     S_DISK_ANGLE = 346,
     S_ENABLED = 347,
     S_MAX_ANGLE = 348,
     S_MIN_ANGLE = 349,
     S_OFFSET = 350,
     S_BBOXSIZE = 351,
     S_BBOXCENTER = 352,
     S_VISIBILITY_LIMIT = 353,
     S_AMBIENT_INTENSITY = 354,
     S_NORMAL = 355,
     S_TEXCOORD = 356,
     S_CCW = 357,
     S_COLOR_PER_VERTEX = 358,
     S_CREASE_ANGLE = 359,
     S_NORMAL_PER_VERTEX = 360,
     S_XDIMENSION = 361,
     S_XSPACING = 362,
     S_ZDIMENSION = 363,
     S_ZSPACING = 364,
     S_BEGIN_CAP = 365,
     S_CROSS_SECTION = 366,
     S_END_CAP = 367,
     S_SPINE = 368,
     S_FOG_TYPE = 369,
     S_VISIBILITY_RANGE = 370,
     S_HORIZONTAL = 371,
     S_JUSTIFY = 372,
     S_LANGUAGE = 373,
     S_LEFT2RIGHT = 374,
     S_TOP2BOTTOM = 375,
     IMAGE_TEXTURE = 376,
     S_SOLID = 377,
     S_KEY = 378,
     S_KEYVALUE = 379,
     S_REPEAT_S = 380,
     S_REPEAT_T = 381,
     S_CONVEX = 382,
     S_BOTTOM = 383,
     S_PICTH = 384,
     S_COORD = 385,
     S_COLOR_INDEX = 386,
     S_COORD_INDEX = 387,
     S_NORMAL_INDEX = 388,
     S_MAX_POSITION = 389,
     S_MIN_POSITION = 390,
     S_ATTENUATION = 391,
     S_APPEARANCE = 392,
     S_GEOMETRY = 393,
     S_DIRECT_OUTPUT = 394,
     S_MUST_EVALUATE = 395,
     S_MAX_BACK = 396,
     S_MIN_BACK = 397,
     S_MAX_FRONT = 398,
     S_MIN_FRONT = 399,
     S_PRIORITY = 400,
     S_SOURCE = 401,
     S_SPATIALIZE = 402,
     S_BERM_WIDTH = 403,
     S_CHOICE = 404,
     S_WHICHCHOICE = 405,
     S_FONTSTYLE = 406,
     S_LENGTH = 407,
     S_MAX_EXTENT = 408,
     S_ROTATION = 409,
     S_SCALE = 410,
     S_CYCLE_INTERVAL = 411,
     S_FIELD_OF_VIEW = 412,
     S_JUMP = 413,
     S_TITLE = 414,
     S_TEXCOORD_INDEX = 415,
     S_HEADLIGHT = 416,
     S_TOP = 417,
     S_BOTTOMRADIUS = 418,
     S_HEIGHT = 419,
     S_POINT = 420,
     S_STRING = 421,
     S_SPACING = 422,
     S_TYPE = 423,
     S_RADIUS = 424,
     S_ON = 425,
     S_INTENSITY = 426,
     S_COLOR = 427,
     S_DIRECTION = 428,
     S_SIZE = 429,
     S_FAMILY = 430,
     S_STYLE = 431,
     S_RANGE = 432,
     S_CENTER = 433,
     S_TRANSLATION = 434,
     S_LEVEL = 435,
     S_DIFFUSECOLOR = 436,
     S_SPECULARCOLOR = 437,
     S_EMISSIVECOLOR = 438,
     S_SHININESS = 439,
     S_TRANSPARENCY = 440,
     S_VECTOR = 441,
     S_POSITION = 442,
     S_ORIENTATION = 443,
     S_LOCATION = 444,
     S_CUTOFFANGLE = 445,
     S_WHICHCHILD = 446,
     S_IMAGE = 447,
     S_SCALEORIENTATION = 448,
     S_DESCRIPTION = 449,
     SFBOOL = 450,
     SFFLOAT = 451,
     SFINT32 = 452,
     SFTIME = 453,
     SFROTATION = 454,
     SFNODE = 455,
     SFCOLOR = 456,
     SFIMAGE = 457,
     SFSTRING = 458,
     SFVEC2F = 459,
     SFVEC3F = 460,
     MFBOOL = 461,
     MFFLOAT = 462,
     MFINT32 = 463,
     MFTIME = 464,
     MFROTATION = 465,
     MFNODE = 466,
     MFCOLOR = 467,
     MFIMAGE = 468,
     MFSTRING = 469,
     MFVEC2F = 470,
     MFVEC3F = 471,
     FIELD = 472,
     EVENTIN = 473,
     EVENTOUT = 474,
     USE = 475,
     S_VALUE_CHANGED = 476
   };
#endif



#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
{

/* Line 2068 of yacc.c  */
#line 20 "src/cybergarage/x3d/VRML97.y"

int		ival;
float	fval;
char	*sval;



/* Line 2068 of yacc.c  */
#line 279 "src/cybergarage/x3d/VRML97.parser.hpp"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif

extern YYSTYPE yylval;


