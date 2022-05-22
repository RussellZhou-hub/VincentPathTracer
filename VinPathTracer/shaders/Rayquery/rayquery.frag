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

#define NUM_SAMPLE 32

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
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
layout (binding = 8, rgba32f) uniform image2D historyColorImages[];
layout (binding = 9, r32f) uniform image2D historyDepthImage;

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

void main() {
    vec3 directColor = vec3(0.0, 0.0, 0.0);
    vec3 indirectColor = vec3(0.0, 0.0, 0.0);
    vec3 surfaceColor=vec3(0.0,0.0,0.0);
    vec3 directAlbedo=vec3(0.0,0.0,0.0);
    vec3 directIr=vec3(0.0,0.0,0.0);
    vec3 inDirectAlbedo=vec3(0.0,0.0,0.0);
    vec3 inDirectIR=vec3(0.0,0.0,0.0);
    vec3 specular=vec3(0.0,0.0,0.0);
    vec3 lightPos=get_Random_QuadArea_Light_Pos(ubo.qLight.A.xyz,  ubo.qLight.B.xyz,  ubo.qLight.C.xyz, ubo.qLight.D.xyz, ubo.frameCount);
    float irradiance=dot(fragNormal,vec3(lightPos-interpolatedPosition));
    irradiance=clamp(irradiance,0.0f,1.0f)/(0.001f*distance(lightPos,interpolatedPosition)+0.01f);
    vec4 diffuseColor;
    int diffuse_id=materialIndexBuffer.data[gl_PrimitiveID].diffuse_idx;
    uint material_id=materialIndexBuffer.data[gl_PrimitiveID].material_id;
    if(diffuse_id==-1){
        diffuseColor=vec4(materialBuffer.data[material_id].diffuse,1.0f);
    }
    else{
        diffuseColor=texture(textures[diffuse_id], fragTexCoord);
    }
    specular=materialBuffer.data[material_id].specular;
    surfaceColor=diffuseColor.xyz;
    directAlbedo=surfaceColor;

    vec4 curClipPos=ubo.proj*ubo.view*vec4(interpolatedPosition,1.0);
    curClipPos.xyz/=curClipPos.w;
    curClipPos.y=-curClipPos.y;
    outWorldPos=curClipPos;

    //outWorldPos=vec4(interpolatedPosition/10000+0.5f,1.0f);
    outNormal=vec4(normalize(fragNormal)/2+0.5,0.0);
    
    vec3 lightColor = vec3(0.6, 0.6, 0.6);
    vec3 geometricNormal=fragNormal;
    vec3 shadowRayOrigin = interpolatedPosition;

    
    vec3 directIr_final=vec3(0.0,0.0,0.0);
    int spp= ubo.mode==5?NUM_SAMPLE:1;
    spp=1;
    float w_sample=1.0/spp;
    for(int i=0;i<spp;i++){
        vec3 lightPosition= spp==1?lightPos:get_Random_QuadArea_Light_Pos(ubo.qLight.A.xyz,  ubo.qLight.B.xyz,  ubo.qLight.C.xyz, ubo.qLight.D.xyz, i,spp);
        vec3 positionToLightDirection = normalize(lightPosition - interpolatedPosition);
        vec3 shadowRayDirection = positionToLightDirection;
        float shadowRayDistance = length(lightPosition - interpolatedPosition) - 0.001f;

        //shadow ray
        rayQueryEXT rayQuery;
        rayQueryInitializeEXT(rayQuery, topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT| gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT, 0xFF, shadowRayOrigin, 0.001f, shadowRayDirection, shadowRayDistance);
  
        while (rayQueryProceedEXT(rayQuery));

        // If the intersection has hit a triangle, the fragment is shadowed
	    if (rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionNoneEXT ) {
            directIr=lightColor* dot(geometricNormal, positionToLightDirection);
            directColor=directIr*directAlbedo;
	    }
        else{
            directColor=vec3(0.0,0.0,0.0);
            directIr=directColor;
        }
        directIr_final+=directIr*w_sample;
    }
    outColor=vec4(directAlbedo,1.0);
    outDirectIr=vec4(directIr_final,1.0);
    
    vec3 lightPosition=lightPos;
    vec3 rayOrigin = interpolatedPosition;
    vec3 rayDirection = getSampledReflectedDirection(ubo.cameraPos.xyz,interpolatedPosition.xyz,geometricNormal,gl_FragCoord.xy,ubo.frameCount);
    vec3 previousNormal = geometricNormal;
    float secondaryRayDistance = length(lightPosition - interpolatedPosition) - 0.001f;

    bool rayActive = true;
    int maxRayDepth = 1;
    if(true){
      for (int rayDepth = 0; rayDepth < maxRayDepth && rayActive; rayDepth++) {
        //secondary ray (or more ray)
        vec3 indirectIr_final=vec3(0.0,0.0,0.0);
        spp= ubo.mode==5?NUM_SAMPLE:1;
        //spp=1;
        w_sample=1.0/spp;
        for(int j=0;j<spp;j++){
            if(spp>1) rayDirection=getUniformSampledSpecularLobeDir(ubo.cameraPos.xyz,interpolatedPosition.xyz,geometricNormal,j,spp);

            rayQueryEXT secondaryRayQuery;
            rayQueryInitializeEXT(secondaryRayQuery, topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, rayOrigin, 0.001f, rayDirection, 1000.0f);

            while (rayQueryProceedEXT(secondaryRayQuery));

            if (rayQueryGetIntersectionTypeEXT(secondaryRayQuery, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
        
                int extensionPrimitiveIndex = rayQueryGetIntersectionPrimitiveIndexEXT(secondaryRayQuery, true);
                vec2 extensionIntersectionBarycentric = rayQueryGetIntersectionBarycentricsEXT(secondaryRayQuery, true);

                ivec3 extensionIndices = ivec3(indexBuffer.data[3 * extensionPrimitiveIndex + 0], indexBuffer.data[3 * extensionPrimitiveIndex + 1], indexBuffer.data[3 * extensionPrimitiveIndex + 2]);
                vec3 extensionBarycentric = vec3(1.0 - extensionIntersectionBarycentric.x - extensionIntersectionBarycentric.y, extensionIntersectionBarycentric.x, extensionIntersectionBarycentric.y);
      
                vec3 extensionVertexA =vertexBuffer.data[extensionIndices.x].pos;
                vec3 extensionVertexB=vertexBuffer.data[extensionIndices.y].pos;
                vec3 extensionVertexC=vertexBuffer.data[extensionIndices.z].pos;

                vec2 extensionVertexA_texCoord =vertexBuffer.data[extensionIndices.x].texCoord;
                vec2 extensionVertexB_texCoord=vertexBuffer.data[extensionIndices.y].texCoord;
                vec2 extensionVertexC_texCoord=vertexBuffer.data[extensionIndices.z].texCoord;
    
                vec3 extensionPosition = extensionVertexA * extensionBarycentric.x + extensionVertexB * extensionBarycentric.y + extensionVertexC * extensionBarycentric.z;
                vec2 extensionTexCoord = extensionVertexA_texCoord * extensionBarycentric.x + extensionVertexB_texCoord * extensionBarycentric.y + extensionVertexC_texCoord * extensionBarycentric.z;
                vec3 extensionNormal = normalize(cross(extensionVertexB - extensionVertexA, extensionVertexC - extensionVertexA));

                vec3 extensionSurfaceColor;
                int extensionDiffuse_id=materialIndexBuffer.data[extensionPrimitiveIndex].diffuse_idx;
                uint extensionMaterial_id=materialIndexBuffer.data[extensionPrimitiveIndex].material_id;
                if(gl_FragCoord.x>1918){
                    debugPrintfEXT("extensionDiffuse_id is %d  extensionMaterial_id is %d \n",extensionDiffuse_id,extensionMaterial_id);
                }
                if(extensionDiffuse_id==-1){
                    extensionSurfaceColor=materialBuffer.data[extensionMaterial_id].diffuse;
                }
                else{
                    extensionSurfaceColor=texture(textures[extensionDiffuse_id], extensionTexCoord).rgb;
                }
                inDirectAlbedo=extensionSurfaceColor;

                //vec2 RayHitPointFragCoord=getFragCoord(extensionPosition.xyz);

            
                //int randomIndex = int(random(gl_FragCoord.xy, ubo.frameCount + rayDepth) * 2 + 40);
                vec3 lightColor = vec3(0.6, 0.6, 0.6);

                vec3 indirectIr_shadow_final=vec3(0.0,0.0,0.0);
                spp= ubo.mode==5?NUM_SAMPLE:1;
                //spp=1;
                w_sample=1.0/spp;
                for(int i=0;i<spp;i++){
                    lightPosition= spp==1?lightPos:get_Random_QuadArea_Light_Pos(ubo.qLight.A.xyz,  ubo.qLight.B.xyz,  ubo.qLight.C.xyz, ubo.qLight.D.xyz, i,spp);
                    vec3 positionToLightDirection = normalize(lightPosition - extensionPosition);

                    vec3 shadowRayOrigin = extensionPosition;
                    vec3 shadowRayDirection = positionToLightDirection;
                    float shadowRayDistance = length(lightPosition - extensionPosition) - 0.001f;
            
                    rayQueryEXT rayQuery;
                    rayQueryInitializeEXT(rayQuery, topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, shadowRayOrigin, 0.001f, shadowRayDirection, shadowRayDistance);
      
                    while (rayQueryProceedEXT(rayQuery));//secondary shadow ray
                    if (rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionNoneEXT) {
                        indirectColor += (1.0 / (rayDepth + 1)) * extensionSurfaceColor * lightColor  * dot(previousNormal, rayDirection) * dot(extensionNormal, positionToLightDirection);
                        inDirectIR=(1.0 / (rayDepth + 1))* lightColor  * dot(previousNormal, rayDirection) * dot(extensionNormal, positionToLightDirection);
                        //indirectColor=extensionSurfaceColor;
                    }
                    else {
                        rayActive = false;
                        inDirectIR=vec3(0.0,0.0,0.0);
                    }
                    indirectIr_shadow_final+=inDirectIR*w_sample;
                }
                inDirectIR=indirectIr_shadow_final;

                //vec3 hemisphere = uniformSampleHemisphere(vec2(random(gl_FragCoord.xy, ubo.frameCount + rayDepth), random(gl_FragCoord.xy, ubo.frameCount + rayDepth + 1)));
                //vec3 alignedHemisphere = alignHemisphereWithCoordinateSystem(hemisphere, extensionNormal);

                //reset rayOrigin...
                rayOrigin = extensionPosition;
                //rayDirection = alignedHemisphere;
                previousNormal = extensionNormal;
                /*
                //RayHitPointFragCoord=getFragCoord(interpolatedPosition.xyz);
                //RayHitPointFragCoord=getFragCoord(extensionPosition.xyz);
                */
                //indirectColor=extensionSurfaceColor;
            }
        
            else {  //secondary ray not hit
                rayActive = false;
                inDirectIR=vec3(0.0,0.0,0.0);
            }
            indirectIr_final+=inDirectIR*w_sample;
        }
        inDirectIR=indirectIr_final;
     }
   }
   
    //outColor = vec4(directColor+indirectColor+diffuseColor.xyz*0.00f,1.0f);
    //outDirectIr=vec4(1.0,0.0,0.0,1.0);
    outIndAlbedo=vec4(inDirectAlbedo,1.0);
    outIndIr=vec4((1-specular)*inDirectIR,1.0);

    //if(isLightSource(materialBuffer.data[material_id].emission)) outColor = vec4(materialBuffer.data[material_id].emission,1.0f);
}