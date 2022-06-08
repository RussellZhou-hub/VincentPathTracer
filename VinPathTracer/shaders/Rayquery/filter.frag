#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_ray_query : enable
#extension GL_EXT_debug_printf : enable
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier: require

#include "../includes/types.glsl"
#include "../includes/random.glsl"
#include "../includes/utils.glsl"

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    mat4 prev_Proj_View;
    QuadArealignt qLight;
    vec4 cameraPos;
    uint frameCount;
    uint mode;  //denoising algorithm   1:raw  2:mvec 3:svgf 4:ours 5: ground truth
} ubo;

layout(binding = 1) uniform sampler2D texSampler;
layout (binding = 2) uniform sampler2D textures[];
layout(binding = 3) buffer MaterialIndexBuffer { Primitive data[]; } materialIndexBuffer;
layout(binding = 4) buffer MaterialBuffer { Material data[]; } materialBuffer;
layout(binding = 5,set=0) uniform accelerationStructureEXT topLevelAS;
layout(binding = 6) buffer VertexBuffer { Vertex data[]; } vertexBuffer;
layout(binding = 7) buffer IndexBuffer { uint data[]; } indexBuffer;
layout (binding = 8, rgba32f) uniform image2D historyColorImages[];  //0:directAlbedo  1:directIR 2:indirectAlbedo 3:indirectIR 4:normal 5:world 6:imageVar
layout (binding = 9, r32f) uniform image2D historyDepthImage;
layout (binding = 10, rgba32f) uniform image2D historyDirectIr;
layout (binding = 11, rgba32f) uniform image2D historyIndIr;
layout (binding = 12, rgba32f) uniform image2D historyIndAlbedo;
layout (binding = 13, rgba32f) uniform image2D historyFinal;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 interpolatedPosition;
layout(location = 3) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outDirectIr;
layout(location = 2) out vec4 outIndAlbedo;
layout(location = 3) out vec4 outIndIr;
layout(location = 4) out vec4 outNormal;
layout(location = 5) out vec4 outWorldPos;
layout(location = 6) out vec4 outDepth;

vec3 fragPos;
bool isShadow=false;
vec2 RayHitPointFragCoord;
vec4 next_itr_Var=vec4(0.0,0.0,0.0,0.0);

float weight(vec2 p,vec2 q);
float w_depth(vec2 p,vec2 q);
float w_normal(vec2 p,vec2 q);
float w_pos(vec2 p,vec2 q);
float w_lumin(vec2 p,vec2 q);
vec4 variance(vec2 p);
vec4 gaussian_filter(vec2 p);
vec4 aTrous_directIr_5_5(vec2 p);
vec4 aTrous_indIr_5_5(vec2 p);
vec4 aTrous_indAlbedo_5_5(vec2 p);
vec4 mvec_Spatial_Filter_directIr(vec2 p);

void main() {
    vec3 directColor = vec3(0.0, 0.0, 0.0);
    vec3 indirectColor = vec3(0.0, 0.0, 0.0);
    vec3 surfaceColor=vec3(0.0,0.0,0.0);
    vec4 direcIr=vec4(0.0,0.0,0.0,0.0);
    vec4 indIr=vec4(0.0,0.0,0.0,0.0);
    //vec4 historyColor=imageLoad(historyColorImages[3],ivec2(gl_FragCoord.xy));
   
    //vec2 myFragCoord=getFragCoord(ubo.proj * ubo.view * ubo.model,interpolatedPosition);
    //if(myFragCoord.x==gl_FragCoord.x) outDirectIr=vec4(0.5,0.0,0.0,1.0);
    //else outDirectIr=vec4(0.0,0.5,0.0,1.0);

    next_itr_Var=vec4(0.0,0.0,variance(gl_FragCoord.xy).z,1.0);
    outColor=next_itr_Var;
    //imageStore(historyColorImages[6],ivec2(gl_FragCoord.xy),3.5*next_itr_Var);

    if(ubo.mode==OURS){
        outIndAlbedo=aTrous_indAlbedo_5_5(gl_FragCoord.xy);
        imageStore(historyIndAlbedo, ivec2(gl_FragCoord.xy),outIndAlbedo);
    }
    else{
        outIndAlbedo=imageLoad(historyIndAlbedo, ivec2(gl_FragCoord.xy));
        imageStore(historyIndAlbedo, ivec2(gl_FragCoord.xy),outIndAlbedo);
    }

    if(ubo.mode==SVGF ||ubo.mode==OURS){
        //direcIr=aTrous_directIr(gl_FragCoord.xy);
        direcIr=aTrous_directIr_5_5(gl_FragCoord.xy);
        indIr=aTrous_indIr_5_5(gl_FragCoord.xy);
        outDirectIr=0.7*direcIr;
        imageStore(historyColorImages[1], ivec2(gl_FragCoord.xy),direcIr);
        imageStore(historyDirectIr, ivec2(gl_FragCoord.xy),direcIr);
        imageStore(historyColorImages[3], ivec2(gl_FragCoord.xy),indIr);
        outIndIr=0.6*indIr;
    }
    else if(ubo.mode==MVEC){  //spatial filter for mvec
        direcIr=mvec_Spatial_Filter_directIr(gl_FragCoord.xy);
        indIr=imageLoad(historyColorImages[3], ivec2(gl_FragCoord.xy));
        outDirectIr=direcIr;
        outIndIr=indIr;
        imageStore(historyDirectIr, ivec2(gl_FragCoord.xy),direcIr);
        imageStore(historyColorImages[3], ivec2(gl_FragCoord.xy),indIr);
    }
    else{
        direcIr=imageLoad(historyDirectIr, ivec2(gl_FragCoord.xy));
        indIr=imageLoad(historyColorImages[3], ivec2(gl_FragCoord.xy));
        outDirectIr=direcIr;
        outIndIr=indIr;
        imageStore(historyDirectIr, ivec2(gl_FragCoord.xy),direcIr);
        imageStore(historyColorImages[3], ivec2(gl_FragCoord.xy),indIr);
        
    }

    //outDirectIr=vec4(0.5,0.0,0.0,1.0);

    //if(isLightSource(materialBuffer.data[material_id].emission)) outColor = vec4(materialBuffer.data[material_id].emission,1.0f);
}

vec4 gaussian_filter(vec2 p){
    mat3 gaussian= mat3(1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0,
                        1.0 / 8.0,  1.0 / 4.0, 1.0 / 8.0,
                        1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0);
    mat3 g_x=mat3(-1,0,1,
                  -1,0,1,
                  -1,0,1);
    mat3 g_y=mat3(-1,-1,-1,
                   0,0,0,
                   1,1,1);
    vec4 sum=vec4(0.0,0.0,0.0,0.0);
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            sum+=gaussian[i][j]*imageLoad(historyDirectIr, ivec2(gl_FragCoord.x+g_x[i][j],gl_FragCoord.y+g_y[i][j]));
        }
    }
    return sum;
}

vec4 mvec_Spatial_Filter_directIr(vec2 p){
    float level=3;
    mat3 gaussian= mat3(1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0,
                        1.0 / 8.0,  1.0 / 4.0, 1.0 / 8.0,
                        1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0);
    mat3 g_x=mat3(-1,0,1,
                  -1,0,1,
                  -1,0,1);
    mat3 g_y=mat3(-1,-1,-1,
                   0,0,0,
                   1,1,1);
    g_x*=level;
    g_y*=level;
    vec4 sum=vec4(0.0,0.0,0.0,0.0);
    vec4 Denominator=vec4(0.0,0.0,0.0,1.0);
    vec4 outFilteredColor;
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            sum+=w_normal(p,ivec2(gl_FragCoord.x+g_x[i][j],gl_FragCoord.y+g_y[i][j]))*gaussian[i][j]*imageLoad(historyDirectIr, ivec2(gl_FragCoord.x+g_x[i][j],gl_FragCoord.y+g_y[i][j]));
            Denominator+=w_normal(p,ivec2(gl_FragCoord.x+g_x[i][j],gl_FragCoord.y+g_y[i][j]))*gaussian[i][j];
        }
    }
    outFilteredColor=sum/Denominator;
    outFilteredColor.w=imageLoad(historyDirectIr, ivec2(gl_FragCoord.xy)).w;
    return outFilteredColor;
}

float w_depth(vec2 p,vec2 q){   //weight of depth in the edge stop function in SVGF
    float sigma_z=1.0;  //1.0 in the SVGF paper
    float epsil=0.01;
    //calculate gradient
    vec2 grad_p;
    float right=imageLoad(historyDepthImage,ivec2(p.x+1,p.y)).x;
    float left=imageLoad(historyDepthImage,ivec2(p.x-1,p.y)).x;
    float up=imageLoad(historyDepthImage,ivec2(p.x,p.y+1)).x;
    float down=imageLoad(historyDepthImage,ivec2(p.x,p.y-1)).x;
    grad_p.x=0.5*right-0.5*left;
    grad_p.y=0.5*up-0.5*down;
    //load p,q
    float depth_p=imageLoad(historyDepthImage,ivec2(p.x,p.y)).x;
    float depth_q=imageLoad(historyDepthImage,ivec2(q.x,q.y)).x;
    //caculate weight
    float weight=exp(-(abs(depth_p-depth_q)/sigma_z*dot(grad_p,(p-q))+epsil));
    return weight;
}

float w_normal(vec2 p,vec2 q){   //weight of normal in the edge stop function in SVGF
    float sigma_n=64;
    vec3 n_p=imageLoad(historyColorImages[4],ivec2(p.xy)).xyz;
    vec3 n_q=imageLoad(historyColorImages[4],ivec2(q.xy)).xyz;
    float weight=pow(max(0,dot(n_p,n_q)),sigma_n);
    return weight;
}

float w_pos(vec2 p,vec2 q){   //weight of pos in the edge stop function add by me
    float sigma_x=0.1;
    float sigma_p=32;
    float epsil=0.0001;
    vec4 x_p=imageLoad(historyColorImages[5],ivec2(p.xy));
    vec4 x_q=imageLoad(historyColorImages[5],ivec2(q.xy));
    float weight=min(1.0f,exp(-distance(x_p.xyz,x_q.xyz)/(sigma_p+epsil)));
    float weight_ID=min(1.0f,exp(-sigma_p*distance(x_p.w,x_q.w)/(sigma_x+epsil)));
    if(ubo.mode==3) return weight;
    if(ubo.mode==4) return weight*weight_ID;
}

float w_lumin(vec2 p,vec2 q){//weight of Luminance in the edge stop function in SVGF
    float sigma_l=100; 
    float epsil=0.00001;
    vec3 lumin_p=gaussian_filter(ivec2(p.xy)).xyz;
    vec3 lumin_q=gaussian_filter(ivec2(q.xy)).xyz;
    float diff=abs(distance(lumin_p,lumin_q)/SQRT_OF_TREE);
    vec4 var=variance(p);
    float weight=exp(-diff/(sigma_l*var.z+epsil));
    return weight;
}

vec4 variance(vec2 p){
    vec2 prevMoments=vec2(0.0f);
    float prevHistoryLen=0.0f;
    vec2 cur_moment;
    float factor=1.0; //ËõÐ¡µ½0~1.0µÄ·¶Î§
    float param_n=0.05;
    float level=16;
    int cnt=0;  //num of the history moment
    float variance_out;
    vec3 normal_p=2*imageLoad(historyColorImages[4],ivec2(p.xy)).xyz-1;
    vec2 prevPos=getFragCoord(ubo.proj*ubo.view,interpolatedPosition);
    if(true /*ubo.frameCount<2 */){
        float first=0.0;
        float second=0.0;
        float tmp;
        float cnt_valid=1.0;
        vec3 curDirectIr=gaussian_filter(ivec2(p.xy)).xyz;
        tmp=length(curDirectIr);
        second+=tmp*tmp;
        first+=tmp;
        vec3 normal_tmp=2*imageLoad(historyColorImages[4],ivec2(p.x+level,p.y)).xyz-1;
        if(length(normal_p-normal_tmp)<param_n){
            tmp=length(imageLoad(historyDirectIr,ivec2(p.x+level,p.y)).xyz);
            second+=tmp*tmp;
            first+=tmp;
            cnt_valid+=1.0;
        }
        normal_tmp=2*imageLoad(historyColorImages[4],ivec2(p.x-level,p.y)).xyz-1;
        if(length(normal_p-normal_tmp)<param_n){
            tmp=length(imageLoad(historyDirectIr,ivec2(p.x-level,p.y)).xyz);
            second+=tmp*tmp;
            first+=tmp;
            cnt_valid+=1.0;
        }
        normal_tmp=2*imageLoad(historyColorImages[4],ivec2(p.x,p.y+level)).xyz-1;
        if(length(normal_p-normal_tmp)<param_n){
            tmp=length(imageLoad(historyDirectIr,ivec2(p.x,p.y+level)).xyz);
            second+=tmp*tmp;
            first+=tmp;
            cnt_valid+=1.0;
        }
        normal_tmp=2*imageLoad(historyColorImages[4],ivec2(p.x,p.y-level)).xyz-1;
        if(length(normal_p-normal_tmp)<param_n){
            tmp=length(imageLoad(historyDirectIr,ivec2(p.x,p.y-level)).xyz);
            second+=tmp*tmp;
            first+=tmp;
            cnt_valid+=1.0;
        }
        first/=cnt_valid;  //first_moment
        second/=cnt_valid;  //second_moment

        prevMoments=vec2(first,second);
        cur_moment=prevMoments;
    }
    else{
        cur_moment.x=length(imageLoad(historyColorImages[1],ivec2(p.xy)).xyz);
        cur_moment.y=cur_moment.x*cur_moment.x;
        
        prevMoments+=imageLoad(historyColorImages[6],ivec2(prevPos.x+1,prevPos.y)).zw/factor;
        cnt++;
        prevMoments+=imageLoad(historyColorImages[6],ivec2(prevPos.x-1,prevPos.y)).zw/factor;
        cnt++;
        prevMoments+=imageLoad(historyColorImages[6],ivec2(prevPos.x,prevPos.y+1)).zw/factor;
        cnt++;
        prevMoments+=imageLoad(historyColorImages[6],ivec2(prevPos.x,prevPos.y-1)).zw/factor;
        cnt++;
    }
    if(cnt>0){
        prevMoments/=cnt;
    }
    float moment_alpha =1.0f/(cnt+1.0);
    //calculate accumulated moments
    float first_Moment=(1-moment_alpha)*prevMoments.x+moment_alpha*cur_moment.x;
    float second_Moment=(1-moment_alpha)*prevMoments.y+moment_alpha*cur_moment.y;
    //outColor.zw=vec2(first_Moment*factor,second_Moment*factor);
    //outColor.xy=outColor.zw;
    variance_out=abs(cur_moment.y-cur_moment.x*cur_moment.x);
    //outColor=vec4(0.0,0.0,variance_out,1.0);
    //next_itr_Var=4*vec4(0.0,0.0,variance_out,1.0);
    //imageStore(historyColorImages[6],ivec2(gl_FragCoord.xy),next_itr_Var);
    variance_out=variance_out>0.0f?1.0f*variance_out:0.0f;
    
    return vec4(first_Moment,second_Moment,variance_out,1.0);
}

float weight(vec2 p,vec2 q){
     return  w_normal(p,q)*w_lumin(p,q)*w_pos(p,q);//w_depth(p,q)*
     //return  w_normal(p,q);
}

vec4 aTrous_directIr_5_5(vec2 p){
    vec4 Numerator=vec4(0.0,0.0,0.0,1.0);
    vec4 Denominator=vec4(0.0,0.0,0.0,1.0);
    vec4 Numerator_var=vec4(0.0,0.0,0.0,1.0);
    float level=1;
    for(int i=-2;i<=2;i++){
        for(int j=-2;j<=2;j++){
            int h_idx=5*(2+j)+2+i;
            float w=weight(gl_FragCoord.xy,vec2(gl_FragCoord.x+i*level,gl_FragCoord.y+j*level));
            Numerator+=h[h_idx]*w*imageLoad(historyDirectIr, ivec2(gl_FragCoord.x+i*level,gl_FragCoord.y+j*level));
            Denominator+=h[h_idx]*w;
            //Numerator_var+=h[h_idx]*h[h_idx]*w*w*variance(vec2(gl_FragCoord.x+i*level,gl_FragCoord.y+j*level));
        }
    }
    Numerator_var/=Denominator*Denominator;
    //next_itr_Var=Numerator_var;

    vec4 outTrous=Numerator/Denominator;
    outTrous.w=imageLoad(historyDirectIr, ivec2(gl_FragCoord.xy)).w;

    return outTrous;
}

vec4 aTrous_indIr_5_5(vec2 p){
    vec4 Numerator=vec4(0.0,0.0,0.0,1.0);
    vec4 Denominator=vec4(0.0,0.0,0.0,1.0);
    float level=1;
    for(int i=-2;i<=2;i++){
        for(int j=-2;j<=2;j++){
            int h_idx=5*(2+j)+2+i;
            float w=weight(gl_FragCoord.xy,vec2(gl_FragCoord.x+i*level,gl_FragCoord.y+j*level));
            Numerator+=h[h_idx]*w*imageLoad(historyIndIr, ivec2(gl_FragCoord.x+i*level,gl_FragCoord.y+j*level));
            Denominator+=h[h_idx]*w;
        }
    }

    vec4 outTrous=Numerator/Denominator;
    outTrous.w=1.0;

    return outTrous;
}

vec4 aTrous_indAlbedo_5_5(vec2 p){
    vec4 Numerator=vec4(0.0,0.0,0.0,1.0);
    vec4 Denominator=vec4(0.0,0.0,0.0,1.0);
    float level=1;
    for(int i=-2;i<=2;i++){
        for(int j=-2;j<=2;j++){
            int h_idx=5*(2+j)+2+i;
            float w=weight(gl_FragCoord.xy,vec2(gl_FragCoord.x+i*level,gl_FragCoord.y+j*level));
            Numerator+=h[h_idx]*w*imageLoad(historyIndAlbedo, ivec2(gl_FragCoord.x+i*level,gl_FragCoord.y+j*level));
            Denominator+=h[h_idx]*w;
        }
    }

    vec4 outTrous=Numerator/Denominator;
    outTrous.w=1.0;

    return outTrous;
}