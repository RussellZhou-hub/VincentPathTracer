#extension GL_EXT_control_flow_attributes : require

#define WIDTH 1920
#define HEIGHT 1080

#define SQRT_OF_TREE 1.7302 

bool isLightSource(vec3 emission){
	if(length(emission)>0.01) return true;
	else return false;
}

vec3 getSampledReflectedDirection(vec3 cameraPos,vec3 inRay,vec3 normal,vec2 uv,float seed){
    inRay=inRay-cameraPos;
    vec3 Ray=reflect(inRay,normal);
    float theta=acos(1-random(uv,seed));  //[0,pi/2]
    float phi=2*M_PI*random(vec2(uv.y,uv.x),seed);
    vec3 RandomRay=vec3(sin(theta)*cos(phi),sin(theta)*sin(phi),cos(theta));
    float weight=0.3;  //reflection rate
    return normalize(weight*Ray+(1-weight)*normalize(RandomRay));
}

vec3 getSampledReflectedDirection(vec3 cameraPos, vec3 inRay, vec3 normal, vec2 uv, float seed,float specularRate) {
    inRay = inRay - cameraPos;
    vec3 Ray = reflect(inRay, normal);
    float theta = acos(1 - random(uv, seed));  //[0,pi/2]
    float phi = 2 * M_PI * random(vec2(uv.y, uv.x), seed);
    vec3 RandomRay = vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
    float weight = specularRate;  //reflection rate
    return normalize(weight * Ray + (1 - weight) * normalize(RandomRay));
}

vec3 getUniformSampledSpecularLobeDir(vec3 cameraPos,vec3 inRay,vec3 normal,int s,int spp){
    inRay=inRay-cameraPos;
    float lobe_range=0.0;
    float weight=0.7;  //reflection rate
    vec3 Ray=normalize(reflect(inRay,normal));

    float Ray_theta=atan(Ray.y/Ray.x);  //[0,pi/2]
    float Ray_phi=atan(sqrt(1-Ray.z*Ray.z)/Ray.z); 

    float Ray_theta_start=Ray_theta-lobe_range*M_PI/2;
    float Ray_phi_start=Ray_phi-lobe_range*2*M_PI;

    //vec3 right=normalize(cross(normal,Ray));
    //vec3 up=normalize(cross(normal,right));

    float stride=1.0f/(sqrt(spp)+0.01f);
    float u=mod(s,sqrt(spp))*stride; // (0,1)
    float v=floor(s/(sqrt(spp)+0.01f))*stride; //(0,1)

    float theta=clamp(Ray_theta_start+u*2*lobe_range*M_PI/2,0.0,M_PI/2);
    float phi=clamp(Ray_phi_start+v*2*lobe_range*2*M_PI,0.0,2*M_PI);
    vec3 RandomRay=vec3(sin(theta)*cos(phi),sin(theta)*sin(phi),cos(theta));

    //Ray=Ray+weight*(right*(u-0.5)+up*(v-0.5));
    
    return normalize(RandomRay);
}

vec2 getFragCoord(mat4 pv,vec3 pos){          //从世界坐标获取对应的上一帧里的屏幕坐标
    vec4 clipPos=pv*vec4(pos,1.0);
      
    clipPos/=clipPos.w;
    clipPos.y=-clipPos.y;
    clipPos.xy+=1;
    clipPos.xy/=2;
    clipPos.x*=WIDTH;
    clipPos.y*=HEIGHT;
    return floor(clipPos.xy)+0.5;
}

float h[25] = float[](1.0 / 256.0, 1.0 / 64.0, 3.0 / 128.0, 1.0 / 64.0, 1.0 / 256.0,
    1.0 / 64.0, 1.0 / 16.0, 3.0 / 32.0, 1.0 / 16.0, 1.0 / 64.0,
    3.0 / 128.0, 3.0 / 32.0, 9.0 / 64.0, 3.0 / 32.0, 3.0 / 128.0,
    1.0 / 64.0, 1.0 / 16.0, 3.0 / 32.0, 1.0 / 16.0, 1.0 / 64.0,
    1.0 / 256.0, 1.0 / 64.0, 3.0 / 128.0, 1.0 / 64.0, 1.0 / 256.0);