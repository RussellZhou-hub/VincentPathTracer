#extension GL_EXT_control_flow_attributes : require

#define WIDTH 1920
#define HEIGHT 1080

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
    float weight=0.5;  //reflection rate
    return normalize(weight*Ray+(1-weight)*normalize(RandomRay));
}

vec3 getUniformSampledSpecularLobeDir(vec3 cameraPos,vec3 inRay,vec3 normal,int s,int spp){
    inRay=inRay-cameraPos;
    float weight=0.5;  //reflection rate
    vec3 Ray=reflect(inRay,normal);
    vec3 right=normalize(cross(normal,Ray));
    vec3 up=normalize(cross(normal,right));

    float stride=1.0f/(sqrt(spp)+0.01f);
    float u=mod(s,sqrt(spp))*stride; // (0,1)
    float v=floor(s/(sqrt(spp)+0.01f))*stride; //(0,1)

    Ray=Ray+weight*(right*(u-0.5)+up*(v-0.5));
    
    return normalize(Ray);
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