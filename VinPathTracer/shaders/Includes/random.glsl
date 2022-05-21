#extension GL_EXT_control_flow_attributes : require

// Random functions from Alan Wolfe's excellent tutorials (https://blog.demofox.org/)
const float pi = 3.14159265359;
const float twopi = 2.0 * pi;

uint wang_hash(inout uint seed)
{
    seed = uint(seed ^ uint(61)) ^ uint(seed >> uint(16));
    seed *= uint(9);
    seed = seed ^ (seed >> 4);
    seed *= uint(0x27d4eb2d);
    seed = seed ^ (seed >> 15);
    return seed;
}

float RandomFloat01(inout uint state)
{
    return float(wang_hash(state)) / 4294967296.0;
}

uint NewRandomSeed(uint v0, uint v1, uint v2)
{
    return uint(v0 * uint(1973) + v1 * uint(9277) + v2 * uint(26699)) | uint(1);
}

// From ray tracing in one weekend (https://raytracing.github.io/)
vec3 RandomInUnitSphere(inout uint seed)
{
	vec3 pos = vec3(0.0);
	do {
		pos = vec3(RandomFloat01(seed), RandomFloat01(seed), RandomFloat01(seed)) * 2.0 - 1.0;
	} while (dot(pos, pos) >= 1.0);
	return pos;
}

float random(vec2 uv, float seed) {     // 0到1的随机数
  return fract(sin(mod(dot(uv, vec2(12.9898, 78.233)) + 1113.1 * seed, pi)) * 43758.5453);
}

float random(vec2 p) 
{ 
    vec2 K1 = vec2(
     23.14069263277926, // e^pi (Gelfond's constant) 
     2.665144142690225 // 2^sqrt(2) (GelfondSchneider constant) 
    ); 
    return fract(cos(dot(p,K1)) * 12345.6789); 
} 

vec3 get_Random_QuadArea_Light_Pos(vec3 A,vec3 B,vec3 C,vec3 D,uint randomIndex){    //参数必须对齐，这里uint参数不能放前面  int需要N，vec3和vec4需要4N 所以vec4不能接在int后面

    vec2 uv = vec2(random(gl_FragCoord.xy, randomIndex), random(vec2(gl_FragCoord.y,gl_FragCoord.x), randomIndex) );

    vec3 mixAB=mix(A,B,uv.x);
    vec3 mixCD=mix(D,D,uv.x);
    vec3 lightPosition=mix(mixAB,mixCD,uv.y);

    return lightPosition;
}

vec3 get_Random_QuadArea_Light_Pos(vec3 A,vec3 B,vec3 C,vec3 D,int s,uint spp){    //ground truth multi-sampling

    float stride=1.0f/(sqrt(spp)+0.01f);
    float u=mod(s,sqrt(spp))*stride;
    float v=floor(s/(sqrt(spp)+0.01f))*stride;
    vec2 uv = vec2(u,v);

    vec3 mixAB=mix(A,B,uv.x);
    vec3 mixCD=mix(D,D,uv.x);
    vec3 lightPosition=mix(mixAB,mixCD,uv.y);

    return lightPosition;
}

