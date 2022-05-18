#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_ray_query : enable
#extension GL_EXT_debug_printf : enable
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier: require

#include "includes/types.glsl"


layout(binding = 1) uniform sampler2D texSampler;
layout (binding = 2) uniform sampler2D textures[];
layout(binding = 3) buffer MaterialIndexBuffer { Primitive data[]; } materialIndexBuffer;
layout(binding = 4) buffer MaterialBuffer { Material data[]; } materialBuffer;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
    vec4 diffuseColor;
    int diffuse_id=materialIndexBuffer.data[gl_PrimitiveID].diffuse_idx;
    uint material_id=materialIndexBuffer.data[gl_PrimitiveID].material_id;
    //diffuse_id=materialBuffer.data[material_id].diffuse_idx;
    if(diffuse_id==-1){
        //diffuseColor=vec4(0.0,1.0,0.0,1.0f);
        diffuseColor=vec4(materialBuffer.data[material_id].diffuse,1.0);
    }
    else{
        diffuseColor=texture(textures[diffuse_id], fragTexCoord);
    }

    //outColor = texture(texSampler, fragTexCoord);
    //outColor = vec4((fragNormal+1)/2,1.0f);
    outColor = diffuseColor;
}