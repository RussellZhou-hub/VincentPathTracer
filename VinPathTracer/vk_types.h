// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vulkan/vulkan.h>
#include<string>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vma/vk_mem_alloc.h>

//we will add our main reusable types here

struct AllocatedBuffer {
    VkBuffer _buffer;
    VmaAllocation _allocation;
};

struct AllocatedImage {
    VkImage _image;
    VmaAllocation _allocation;
};

struct QuadArealignt {
    alignas(4) glm::vec4 A;  //    A* * *B
    alignas(4) glm::vec4 B;  //    *     *
    alignas(4) glm::vec4 C;  //    *     *
    alignas(4) glm::vec4 D;  //    C* * *D
};

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::mat4 prev_Proj_View;
    alignas(16) QuadArealignt quadArealignt;
    alignas(4) glm::vec4 cameraPos;
    alignas(4) uint32_t frameCount;
    alignas(4) uint32_t mode;  //denoising algorithm
    alignas(4) uint32_t samples;  //sampling rate for GT
};

struct Texture {
    std::string fileName;
    VkImage textureImage;
    VkDeviceMemory textureImageMemory;
    VkImageView textureImageView;
    VkSampler textureSampler;
};

struct Attachment {
    std::string name;
    VkImage image;
    VkDeviceMemory imageMemory;
    VkImageView imageView;
    VkFormat format;
};

struct Pipeline {
    VkRenderPass renderPass;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;
};

struct Primitive {
    uint32_t material_id = 0;
    int diffuse_idx;
    static const int vertices_count = 3;//hardcorded 3
};

struct Material {
    alignas(4) glm::vec3 ambient; int padA;
    alignas(4) glm::vec3 diffuse; int padB;
    alignas(4) glm::vec3 specular; int padC;
    alignas(4) glm::vec3 emission; //int padD=1.0;
    int diffuse_idx;
};
