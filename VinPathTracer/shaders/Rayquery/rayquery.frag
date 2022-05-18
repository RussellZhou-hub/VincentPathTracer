#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_ray_query : enable
#extension GL_EXT_debug_printf : enable
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier: require

#include "../includes/types.glsl"
#include "../includes/random.glsl"

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    QuadArealignt qLight;
    uint frameCount;
} ubo;

layout(binding = 1) uniform sampler2D texSampler;
layout (binding = 2) uniform sampler2D textures[];
layout(binding = 3) buffer MaterialIndexBuffer { Primitive data[]; } materialIndexBuffer;
layout(binding = 4) buffer MaterialBuffer { Material data[]; } materialBuffer;
layout(binding = 5) uniform accelerationStructureEXT topLevelAS;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 interpolatedPosition;
layout(location = 3) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 directColor = vec3(0.0, 0.0, 0.0);
    vec3 indirectColor = vec3(0.0, 0.0, 0.0);
    vec3 surfaceColor=vec3(0.0,0.0,0.0);
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
    surfaceColor=diffuseColor.xyz;
    
    vec3 lightColor = vec3(0.6, 0.6, 0.6);
    vec3 lightPosition=lightPos;
    vec3 geometricNormal=fragNormal;

    vec3 positionToLightDirection = normalize(lightPosition - interpolatedPosition);

    vec3 shadowRayOrigin = interpolatedPosition;
    vec3 shadowRayDirection = positionToLightDirection;
    float shadowRayDistance = length(lightPosition - interpolatedPosition) - 0.001f;

    
    //shadow ray
    rayQueryEXT rayQuery;
    rayQueryInitializeEXT(rayQuery, topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, shadowRayOrigin, 0.001f, shadowRayDirection, shadowRayDistance);
  
    while (rayQueryProceedEXT(rayQuery));

    if (rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionNoneEXT) {
        directColor=surfaceColor * lightColor * dot(geometricNormal, positionToLightDirection); 
    }
    else {  //now in shadow
        directColor=vec3(0.0,0.0,0.0);
    }

    outColor = vec4(directColor+diffuseColor.xyz*0.00f,1.0f);
}