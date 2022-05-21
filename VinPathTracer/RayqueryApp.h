#pragma once
#define GLFW_INCLUDE_VULKAN
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <array>
#include <optional>
#include <set>
#include <unordered_map>
#include"VkApp.h"
#include "VulkanBuffer.h"

class RayQueryApp : VkApplication {
public:
	RayQueryApp();
	void run();

	

	void initVulkan();
	void mainLoop();
	void drawFrame();
	void addRayQueryExtension();
	void setModelPath(std::string path);
	void createLogicalDevice();
	void prepare();
	void createDescriptorPool();
	void createDescriptorSetLayout();
	void createDescriptorSets();
	void createAttachments();
	void createFramebuffers();
	void createGraphicsPipeline();
	void createGraphicsPipeline(Pipeline& pipeline, std::string vsName, std::string fsName);
	void createRenderPass();
	void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);
	uint64_t getBufferDeviceAddress(VkBuffer buffer);
	void createBottomLevelAccelerationStructure();
	void createTopLevelAccelerationStructure();

	//attachment stuff
	std::vector<Attachment> inPutAttachments;
	std::vector<Attachment> outPutAttachments;

	Attachment historyDepth;

	Pipeline pipline_filter;

	// Function pointers for ray tracing related stuff
	PFN_vkGetBufferDeviceAddressKHR vkGetBufferDeviceAddressKHR;
	PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR;
	PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR;
	PFN_vkGetAccelerationStructureBuildSizesKHR vkGetAccelerationStructureBuildSizesKHR;
	PFN_vkGetAccelerationStructureDeviceAddressKHR vkGetAccelerationStructureDeviceAddressKHR;
	PFN_vkBuildAccelerationStructuresKHR vkBuildAccelerationStructuresKHR;
	PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR;
	PFN_vkCmdTraceRaysKHR vkCmdTraceRaysKHR;
	PFN_vkGetRayTracingShaderGroupHandlesKHR vkGetRayTracingShaderGroupHandlesKHR;
	PFN_vkCreateRayTracingPipelinesKHR vkCreateRayTracingPipelinesKHR;

	// Available features and properties
	VkPhysicalDeviceRayTracingPipelinePropertiesKHR  rayTracingPipelineProperties{};
	VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures{};

	// Enabled features and properties
	VkPhysicalDeviceBufferDeviceAddressFeatures enabledBufferDeviceAddresFeatures{};
	VkPhysicalDeviceRayTracingPipelineFeaturesKHR enabledRayTracingPipelineFeatures{};
	VkPhysicalDeviceAccelerationStructureFeaturesKHR enabledAccelerationStructureFeatures{};

	// Holds information for a ray tracing scratch buffer that is used as a temporary storage
	struct ScratchBuffer
	{
		uint64_t deviceAddress = 0;
		VkBuffer handle = VK_NULL_HANDLE;
		VkDeviceMemory memory = VK_NULL_HANDLE;
	};

	// Holds information for a ray tracing acceleration structure
	struct AccelerationStructure {
		VkAccelerationStructureKHR handle;
		uint64_t deviceAddress = 0;
		VkDeviceMemory memory;
		VkBuffer buffer;
	};

	// Holds information for a storage image that the ray tracing shaders output to
	struct StorageImage {
		VkDeviceMemory memory = VK_NULL_HANDLE;
		VkImage image = VK_NULL_HANDLE;
		VkImageView view = VK_NULL_HANDLE;
		VkFormat format;
	} storageImage;

	// Extends the buffer class and holds information for a shader binding table
	class ShaderBindingTable : public vks::Buffer {
	public:
		VkStridedDeviceAddressRegionKHR stridedDeviceAddressRegion{};
	};

	// Set to true, to denote that the sample only uses ray queries (changes extension and render pass handling)
	bool rayQueryOnly = false;
	void createAccelerationStructure(AccelerationStructure& accelerationStructure, VkAccelerationStructureTypeKHR type, VkAccelerationStructureBuildSizesInfoKHR buildSizeInfo);
	void deleteAccelerationStructure(AccelerationStructure& accelerationStructure);
	ScratchBuffer createScratchBuffer(VkDeviceSize size);
	void deleteScratchBuffer(ScratchBuffer& scratchBuffer);

	AccelerationStructure bottomLevelAS{};
	AccelerationStructure topLevelAS{};

	VkAccelerationStructureKHR accelerationStructure;
	VkBuffer accelerationStructureBuffer;
	VkDeviceMemory accelerationStructureBufferMemory;

	VkAccelerationStructureKHR bottomLevelAccelerationStructure;
	VkBuffer bottomLevelAccelerationStructureBuffer;
	VkDeviceMemory bottomLevelAccelerationStructureBufferMemory;

	VkAccelerationStructureKHR topLevelAccelerationStructure;
	VkBuffer topLevelAccelerationStructureBuffer;
	VkDeviceMemory topLevelAccelerationStructureBufferMemory;
};