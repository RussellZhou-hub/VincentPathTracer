#include "RayqueryApp.h"
#include "vk_initializers.h"
#include"VulkanDevice.h"
#include"VulkanTools.h"
#include "Utils.h"
//#include "stb_image.h"
//#include"VkApp.h"

#define VK_CHECK_RESULT(f)																				\
{																										\
	VkResult res = (f);																					\
	if (res != VK_SUCCESS)																				\
	{																									\
		/*std::cout << "Fatal : VkResult is \"" << vks::tools::errorString(res) << "\" in " << __FILE__ << " at line " << __LINE__ << "\n"; \*/\
		assert(res == VK_SUCCESS);																		\
	}																									\
}

RayQueryApp::RayQueryApp()
{
	addRayQueryExtension();
    setModelPath("sponza");
    //setModelPath("bathroom-blender"); //camera.pos=vec3(11.19,9.25,20.89);
    
    setShaderFileName("Rayquery/rayquery.vert.spv", "Rayquery/rayquery.frag.spv");
    //setShaderFileName("basic.vert.spv", "basic.frag.spv");
}

void RayQueryApp::run()
{
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
}

void RayQueryApp::initVulkan()
{
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createCommandPool();
    loadModel();
    createRenderPass();
    createDescriptorSetLayout();
    createGraphicsPipeline();
    pipline_filter.renderPass = renderPass;
    pipline_filter_2nd.renderPass = renderPass;
    pipline_filter_3rd.renderPass = renderPass;
    createGraphicsPipeline(pipline_filter, "Rayquery/rayquery.vert.spv", "Rayquery/filter.frag.spv");
    createGraphicsPipeline(pipline_filter_2nd, "Rayquery/rayquery.vert.spv", "Rayquery/filter_2nd.frag.spv");
    createGraphicsPipeline(pipline_filter_3rd, "Rayquery/rayquery.vert.spv", "Rayquery/filter_3rd.frag.spv");
    createDepthResources();
    createFramebuffers();
    createTextureImage();
    createTextureImageView();
    createTextureSampler();
    createVertexBuffer();
    createIndexBuffer();
    createMaterialsBuffer();
    createUniformBuffers();
    prepare();  //prepare ray tracing(ray query) stuff
    createBottomLevelAccelerationStructure();
    createTopLevelAccelerationStructure();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
    createSyncObjects();
}

void RayQueryApp::mainLoop()
{
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        {
            setIcon();

            if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
                glfwSetWindowShouldClose(window, true);

            glfwSetInputMode(window, GLFW_STICKY_MOUSE_BUTTONS, GLFW_CURSOR_DISABLED);
            //glfwSetCursorPosCallback(window, (GLFWcursorposfun)mouse_callback);
            double xpos, ypos;
            glfwGetCursorPos(window, &xpos, &ypos);
            mouse_callback(window, xpos, ypos);
            glfwSetScrollCallback(window, scroll_callback);
            glfwSetMouseButtonCallback(window, mouse_button_callback);
            scroll_process();

            const float cameraSpeed = 100.0f * camera.getDeltaTime(glfwGetTime()); // adjust accordingly
            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
                camera.pos += cameraSpeed * camera.front;
                ubo.frameCount = 1; //camera moved
            }
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
                camera.pos -= cameraSpeed * camera.front;
                ubo.frameCount = 1; //camera moved
            }

            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
                camera.pos -= glm::normalize(glm::cross(camera.front, camera.up)) * cameraSpeed;
                ubo.frameCount = 1; //camera moved
            }

            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
                camera.pos += glm::normalize(glm::cross(camera.front, camera.up)) * cameraSpeed;
                ubo.frameCount = 1; //camera moved
            }

            // adjust denoising mode
            if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) ubo.mode = 1;
            if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) ubo.mode = 2;
            if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) ubo.mode = 3;
            if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS) ubo.mode = 4;
            if (glfwGetKey(window, GLFW_KEY_5) == GLFW_PRESS) ubo.mode = 5;
        }
        drawFrame();
    }

    vkDeviceWaitIdle(device);
}


void RayQueryApp::drawFrame()
{
    vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapChain();
        return;
    }
    else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("failed to acquire swap chain image!");
    }

    updateUniformBuffer(currentFrame);

    vkResetFences(device, 1, &inFlightFences[currentFrame]);

    vkResetCommandBuffer(commandBuffers[currentFrame], /*VkCommandBufferResetFlagBits*/ 0);
    recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

    VkSubmitInfo submitInfo = vkinit::submit_info(&commandBuffers[currentFrame]);

    VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
    VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;

    VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
        throw std::runtime_error("failed to submit draw command buffer!");
    }

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;

    VkSwapchainKHR swapChains[] = { swapChain };
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;

    presentInfo.pImageIndices = &imageIndex;

    result = vkQueuePresentKHR(presentQueue, &presentInfo);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
        framebufferResized = false;
        recreateSwapChain();
    }
    else if (result != VK_SUCCESS) {
        throw std::runtime_error("failed to present swap chain image!");
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void RayQueryApp::addRayQueryExtension()
{
	deviceExtensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
	deviceExtensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
	deviceExtensions.push_back(VK_KHR_SPIRV_1_4_EXTENSION_NAME);
	deviceExtensions.push_back(VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME);
	deviceExtensions.push_back(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
	deviceExtensions.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
	deviceExtensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
	deviceExtensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
	deviceExtensions.push_back(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
	deviceExtensions.push_back(VK_KHR_MAINTENANCE_3_EXTENSION_NAME);
	deviceExtensions.push_back(VK_KHR_MAINTENANCE_1_EXTENSION_NAME);
	deviceExtensions.push_back(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);
    deviceExtensions.push_back(VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
}

void RayQueryApp::setModelPath(std::string path)
{
    model_Path = path;
}

void RayQueryApp::createLogicalDevice()
{
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::vector<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() ,indices.computeFamily.value() };

    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo = vkinit::device_Queue_create_info(queueFamily, &queuePriority);
        queueCreateInfos.push_back(queueCreateInfo);
    }

    VkPhysicalDeviceBufferDeviceAddressFeaturesEXT bufferDeviceAddressFeatures = vkinit::bufferDeviceAddress_features();
    VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures = vkinit::rayQuery_features(&bufferDeviceAddressFeatures);
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures = vkinit::accelerationStructure_features(&rayQueryFeatures);

    VkPhysicalDeviceFeatures deviceFeatures{};
    deviceFeatures.samplerAnisotropy = VK_TRUE;

    VkDeviceCreateInfo createInfo = vkinit::device_create_info(&accelerationStructureFeatures, static_cast<uint32_t>(queueCreateInfos.size()), queueCreateInfos.data(),
        static_cast<uint32_t>(deviceExtensions.size()), deviceExtensions.data(), &deviceFeatures);
    if (enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    }
    else {
        createInfo.enabledLayerCount = 0;
    }

    VK_CHECK(vkCreateDevice(physicalDevice, &createInfo, nullptr, &device), "failed to create logical device!");

    vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
    vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
    vkGetDeviceQueue(device, indices.computeFamily.value(), 0, &computeQueue);
}

void RayQueryApp::prepare()
{
    // Get properties and features
    rayTracingPipelineProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
    VkPhysicalDeviceProperties2 deviceProperties2{};
    deviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    deviceProperties2.pNext = &rayTracingPipelineProperties;
    vkGetPhysicalDeviceProperties2(physicalDevice, &deviceProperties2);
    accelerationStructureFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    VkPhysicalDeviceFeatures2 deviceFeatures2{};
    deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    deviceFeatures2.pNext = &accelerationStructureFeatures;
    vkGetPhysicalDeviceFeatures2(physicalDevice, &deviceFeatures2);
    // Get the function pointers required for ray tracing
    vkGetBufferDeviceAddressKHR = reinterpret_cast<PFN_vkGetBufferDeviceAddressKHR>(vkGetDeviceProcAddr(device, "vkGetBufferDeviceAddressKHR"));
    vkCmdBuildAccelerationStructuresKHR = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR"));
    vkBuildAccelerationStructuresKHR = reinterpret_cast<PFN_vkBuildAccelerationStructuresKHR>(vkGetDeviceProcAddr(device, "vkBuildAccelerationStructuresKHR"));
    vkCreateAccelerationStructureKHR = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR"));
    vkDestroyAccelerationStructureKHR = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(vkGetDeviceProcAddr(device, "vkDestroyAccelerationStructureKHR"));
    vkGetAccelerationStructureBuildSizesKHR = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR"));
    vkGetAccelerationStructureDeviceAddressKHR = reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(vkGetDeviceProcAddr(device, "vkGetAccelerationStructureDeviceAddressKHR"));
    vkCmdTraceRaysKHR = reinterpret_cast<PFN_vkCmdTraceRaysKHR>(vkGetDeviceProcAddr(device, "vkCmdTraceRaysKHR"));
    vkGetRayTracingShaderGroupHandlesKHR = reinterpret_cast<PFN_vkGetRayTracingShaderGroupHandlesKHR>(vkGetDeviceProcAddr(device, "vkGetRayTracingShaderGroupHandlesKHR"));
    vkCreateRayTracingPipelinesKHR = reinterpret_cast<PFN_vkCreateRayTracingPipelinesKHR>(vkGetDeviceProcAddr(device, "vkCreateRayTracingPipelinesKHR"));
}

void RayQueryApp::createDescriptorPool()
{
    std::vector<VkDescriptorPoolSize> poolSizes{};
    poolSizes.resize(2);
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
    poolSizes.push_back(vkinit::des_pool_size(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT)));
    poolSizes.push_back(vkinit::des_pool_size(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT)));


    VkDescriptorPoolCreateInfo poolInfo = vkinit::descriptorPool_create_info(static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT), static_cast<uint32_t>(poolSizes.size()), poolSizes.data());
    VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool), "failed to create descriptor pool!");
}

void RayQueryApp::createDescriptorSetLayout()
{
    VkDescriptorSetLayoutBinding uboLayoutBinding = vkinit::descriptorSet_layout_bindings(0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
    VkDescriptorSetLayoutBinding samplerLayoutBinding = vkinit::descriptorSet_layout_bindings(1, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT);

    std::vector<VkDescriptorSetLayoutBinding> bindings = { uboLayoutBinding, samplerLayoutBinding };
    bindings.resize(2);
    if (textures.size() > 0) {
        //bindings.resize(bindings.size() + 1);
        VkDescriptorSetLayoutBinding binding = vkinit::descriptorSet_layout_bindings(bindings.size(), textures.size(), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT);
        bindings.push_back(binding);//Texture array
    }
    bindings.push_back(vkinit::descriptorSet_layout_bindings(bindings.size(), 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT));
    bindings.push_back(vkinit::descriptorSet_layout_bindings(bindings.size(), 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT));
    bindings.push_back(vkinit::descriptorSet_layout_bindings(bindings.size(), 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_FRAGMENT_BIT));
    bindings.push_back(vkinit::descriptorSet_layout_bindings(bindings.size(), 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT));//vertex buffer
    bindings.push_back(vkinit::descriptorSet_layout_bindings(bindings.size(), 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT));//index buffer
    VkDescriptorSetLayoutBinding history_image_binding = vkinit::descriptorSet_layout_bindings(bindings.size(), inPutAttachments.size(), VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_FRAGMENT_BIT);
    bindings.push_back(history_image_binding);//History color image array
    bindings.push_back(vkinit::descriptorSet_layout_bindings(bindings.size(), 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_FRAGMENT_BIT));//History depth image

    VkDescriptorSetLayoutCreateInfo layoutInfo = vkinit::descriptorSetLayout_create_info(static_cast<uint32_t>(bindings.size()), bindings.data());
    VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout), "failed to create descriptor set layout!");
}

void RayQueryApp::createDescriptorSets()
{
    std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo = vkinit::descriptorSet_allocate_info(descriptorPool, static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT), layouts.data());
    descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
    VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()), "failed to allocate descriptor sets!");

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        VkDescriptorBufferInfo bufferInfo = vkinit::buffer_info(uniformBuffers[i], 0, sizeof(UniformBufferObject));
        VkDescriptorImageInfo imageInfo = vkinit::image_info(textureImageView, textureSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        std::vector<VkWriteDescriptorSet> descriptorWrites{};
        descriptorWrites.resize(2);

        descriptorWrites[0] = vkinit::writeDescriptorSets_info(nullptr, descriptorSets[i], 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &bufferInfo);
        descriptorWrites[1] = vkinit::writeDescriptorSets_info(nullptr, descriptorSets[i], 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &imageInfo);
        if (textures.size() > 0) {
            VkDescriptorImageInfo* imageInfo_tex_array;
            imageInfo_tex_array = vkinit::get_textures_descriptor_ImageInfos(textures.size(), textures);
            descriptorWrites.push_back(vkinit::writeDescriptorSets_info(nullptr, descriptorSets[i], descriptorWrites.size(), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, imageInfo_tex_array, nullptr, nullptr, textures.size()));
        }
        VkDescriptorBufferInfo materialIndexBufferInfo = vkinit::buffer_info(materialIndexBuffer);
        descriptorWrites.push_back(vkinit::writeDescriptorSets_info(nullptr, descriptorSets[i], descriptorWrites.size(), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &materialIndexBufferInfo));
        VkDescriptorBufferInfo materialBufferInfo = vkinit::buffer_info(materialBuffer);
        descriptorWrites.push_back(vkinit::writeDescriptorSets_info(nullptr, descriptorSets[i], descriptorWrites.size(), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &materialBufferInfo));
        VkWriteDescriptorSetAccelerationStructureKHR descriptorSetAccelerationStructure = vkinit::descriptorSetAS_info(&topLevelAS.handle);
        descriptorWrites.push_back(vkinit::writeDescriptorSets_info(&descriptorSetAccelerationStructure, descriptorSets[i], descriptorWrites.size(), VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR));
        VkDescriptorBufferInfo vertexBufferInfo = vkinit::buffer_info(vertexBuffer);
        descriptorWrites.push_back(vkinit::writeDescriptorSets_info(nullptr, descriptorSets[i], descriptorWrites.size(), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &vertexBufferInfo));
        VkDescriptorBufferInfo indexBufferInfo = vkinit::buffer_info(indexBuffer);
        descriptorWrites.push_back(vkinit::writeDescriptorSets_info(nullptr, descriptorSets[i], descriptorWrites.size(), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &indexBufferInfo));
        VkDescriptorImageInfo* imageInfo_history_array = vkinit::get_inputAttach_descriptor_ImageInfos(inPutAttachments.size(), inPutAttachments);
        descriptorWrites.push_back(vkinit::writeDescriptorSets_info(nullptr, descriptorSets[i], descriptorWrites.size(), VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, imageInfo_history_array, nullptr, nullptr, inPutAttachments.size()));
        VkDescriptorImageInfo depthImageInfo = vkinit::image_info(historyDepth.imageView, 0, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        descriptorWrites.push_back(vkinit::writeDescriptorSets_info(nullptr, descriptorSets[i], descriptorWrites.size(), VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &depthImageInfo));
        

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void RayQueryApp::createAttachments()  //create color attachments not including depth attachment
{
    for (auto i = 0; i < 6; i++) {
        Attachment attach;
        attach.format = swapChainImageFormat;
        createImage(WIDTH, HEIGHT, attach.format, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, attach.image , attach.imageMemory);
        attach.imageView=createImageView(attach.image, attach.format, VK_IMAGE_ASPECT_COLOR_BIT);
        VkImageMemoryBarrier imageMemoryBarrier=vkinit::barrier_des(attach.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        VkCommandBuffer commandBuffer=beginSingleTimeCommands();
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &imageMemoryBarrier);
        endSingleTimeCommands(commandBuffer);
        outPutAttachments.push_back(attach);

        Attachment attach_input;
        attach_input.format = swapChainImageFormat;
        createImage(WIDTH, HEIGHT, attach_input.format, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, attach_input.image, attach_input.imageMemory);
        attach_input.imageView = createImageView(attach_input.image, attach_input.format, VK_IMAGE_ASPECT_COLOR_BIT);
        inPutAttachments.push_back(attach_input);
    }
    Attachment historyVariance;
    historyVariance.format = swapChainImageFormat;
    createImage(WIDTH, HEIGHT, historyVariance.format, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, historyVariance.image, historyVariance.imageMemory);
    historyVariance.imageView = createImageView(historyVariance.image, historyVariance.format, VK_IMAGE_ASPECT_COLOR_BIT);
    inPutAttachments.push_back(historyVariance);

    historyDepth.format = findDepthFormat();
    createImage(WIDTH, HEIGHT, historyDepth.format, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, historyDepth.image, historyDepth.imageMemory);
    historyDepth.imageView = createImageView(historyDepth.image, historyDepth.format, VK_IMAGE_ASPECT_DEPTH_BIT);
}

void RayQueryApp::createFramebuffers()
{
    swapChainFramebuffers.resize(swapChainImageViews.size());

    for (size_t i = 0; i < swapChainImageViews.size(); i++) {
        std::vector<VkImageView> attachments;
        attachments.push_back(swapChainImageViews[i]);
        for (auto i = 1; i < outPutAttachments.size(); i++) {
            attachments.push_back(outPutAttachments[i].imageView);
        }
        attachments.push_back(depthImageView);

        VkFramebufferCreateInfo framebufferInfo = vkinit::framebuffer_create_info(renderPass, static_cast<uint32_t>(attachments.size()), attachments.data(), swapChainExtent.width, swapChainExtent.height);

        if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create framebuffer!");
        }
    }
}

void RayQueryApp::createGraphicsPipeline()
{
#ifdef _DEBUG
    std::string baseDir("C:/Users/Rocki/source/repos/VinPathTracer/VinPathTracer/shaders/");
#else
    std::string baseDir("shaders/");
#endif
    auto vertShaderCode = readFile(baseDir + vertShaderName);
    auto fragShaderCode = readFile(baseDir + fragShaderName);


    VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

    VkPipelineShaderStageCreateInfo vertShaderStageInfo = vkinit::ShaderStage_info(VK_SHADER_STAGE_VERTEX_BIT, vertShaderModule);
    VkPipelineShaderStageCreateInfo fragShaderStageInfo = vkinit::ShaderStage_info(VK_SHADER_STAGE_FRAGMENT_BIT, fragShaderModule);
    VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();
    VkPipelineVertexInputStateCreateInfo vertexInputInfo = vkinit::vertexInputState_create_info(&bindingDescription, attributeDescriptions.data(), 1, static_cast<uint32_t>(attributeDescriptions.size()));

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = vkinit::inputAssembly_create_info(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_FALSE);

    VkViewport viewport = vkinit::viewport_des(0.0f, 0.0f, (float)swapChainExtent.width, (float)swapChainExtent.height, 0.0f, 1.0f);
    VkRect2D scissor = vkinit::scissor({ 0, 0 }, swapChainExtent);

    VkPipelineViewportStateCreateInfo viewportState = vkinit::viewportState_create_info(&viewport, &scissor);
    VkPipelineRasterizationStateCreateInfo rasterizer = vkinit::rasterizationState_create_info();
    VkPipelineMultisampleStateCreateInfo multisampling = vkinit::multisampleState_create_info();
    VkPipelineDepthStencilStateCreateInfo depthStencil = vkinit::depthStencil_create_info();

    VkPipelineColorBlendAttachmentState* pColorBlendAttachmentState = (VkPipelineColorBlendAttachmentState*)malloc(outPutAttachments.size() * sizeof(VkPipelineColorBlendAttachmentState));
    for (auto i = 0; i < outPutAttachments.size(); i++) {
        pColorBlendAttachmentState[i].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        pColorBlendAttachmentState[i].blendEnable = VK_FALSE;
    }

    float blendConstants[4] = { 0.0,0.0,0.0,0.0 };
    VkPipelineColorBlendStateCreateInfo colorBlending = vkinit::colorBlendState_create_info(outPutAttachments.size(), pColorBlendAttachmentState, blendConstants);
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = vkinit::pipelineLayout_create_info(1, &descriptorSetLayout);

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create pipeline layout!");
    }

    VkGraphicsPipelineCreateInfo pipelineInfo = vkinit::graphicsPipeline_create_info(2, shaderStages, &vertexInputInfo, &inputAssembly, &viewportState,
        &rasterizer, &multisampling, &depthStencil, &colorBlending, pipelineLayout, renderPass, 0);

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
}

void RayQueryApp::createGraphicsPipeline(Pipeline& pipeline, std::string vsName, std::string fsName)
{
#ifdef _DEBUG
    std::string baseDir("C:/Users/Rocki/source/repos/VinPathTracer/VinPathTracer/shaders/");
#else
    std::string baseDir("shaders/");
#endif
    auto vertShaderCode = readFile(baseDir + vsName);
    auto fragShaderCode = readFile(baseDir + fsName);


    VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

    VkPipelineShaderStageCreateInfo vertShaderStageInfo = vkinit::ShaderStage_info(VK_SHADER_STAGE_VERTEX_BIT, vertShaderModule);
    VkPipelineShaderStageCreateInfo fragShaderStageInfo = vkinit::ShaderStage_info(VK_SHADER_STAGE_FRAGMENT_BIT, fragShaderModule);
    VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();
    VkPipelineVertexInputStateCreateInfo vertexInputInfo = vkinit::vertexInputState_create_info(&bindingDescription, attributeDescriptions.data(), 1, static_cast<uint32_t>(attributeDescriptions.size()));

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = vkinit::inputAssembly_create_info(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_FALSE);

    VkViewport viewport = vkinit::viewport_des(0.0f, 0.0f, (float)swapChainExtent.width, (float)swapChainExtent.height, 0.0f, 1.0f);
    VkRect2D scissor = vkinit::scissor({ 0, 0 }, swapChainExtent);

    VkPipelineViewportStateCreateInfo viewportState = vkinit::viewportState_create_info(&viewport, &scissor);
    VkPipelineRasterizationStateCreateInfo rasterizer = vkinit::rasterizationState_create_info();
    VkPipelineMultisampleStateCreateInfo multisampling = vkinit::multisampleState_create_info();
    VkPipelineDepthStencilStateCreateInfo depthStencil = vkinit::depthStencil_create_info();

    VkPipelineColorBlendAttachmentState* pColorBlendAttachmentState = (VkPipelineColorBlendAttachmentState*)malloc(outPutAttachments.size() * sizeof(VkPipelineColorBlendAttachmentState));
    for (auto i = 0; i < outPutAttachments.size(); i++) {
        pColorBlendAttachmentState[i].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        pColorBlendAttachmentState[i].blendEnable = VK_FALSE;
    }

    float blendConstants[4] = { 0.0,0.0,0.0,0.0 };
    VkPipelineColorBlendStateCreateInfo colorBlending = vkinit::colorBlendState_create_info(outPutAttachments.size(), pColorBlendAttachmentState, blendConstants);
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = vkinit::pipelineLayout_create_info(1, &descriptorSetLayout);

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipeline.pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create pipeline layout!");
    }

    VkGraphicsPipelineCreateInfo pipelineInfo = vkinit::graphicsPipeline_create_info(2, shaderStages, &vertexInputInfo, &inputAssembly, &viewportState,
        &rasterizer, &multisampling, &depthStencil, &colorBlending, pipeline.pipelineLayout, pipeline.renderPass, 0);

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline.graphicsPipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
}

void RayQueryApp::createRenderPass()
{
    VkAttachmentDescription colorAttachment = vkinit::colorAttachment_des(swapChainImageFormat, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    VkAttachmentDescription colorAttachment_2 = vkinit::colorAttachment_des(swapChainImageFormat, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkAttachmentDescription depthAttachment = vkinit::depthAttachment_des(findDepthFormat());

    createAttachments();
    VkAttachmentReference* colorAttachmentRef = (VkAttachmentReference*)malloc(outPutAttachments.size() * sizeof(VkAttachmentReference));
    for (auto i = 0; i < outPutAttachments.size(); i++) {
        colorAttachmentRef[i].attachment = i;  //location in fragment shader
        colorAttachmentRef[i].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    };

    VkAttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = outPutAttachments.size();
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass = vkinit::subpass_des(outPutAttachments.size(), colorAttachmentRef, &depthAttachmentRef);

    VkSubpassDependency dependency = vkinit::dependency_des(0);

    std::vector<VkAttachmentDescription> attachments;
    for (auto i = 0; i < outPutAttachments.size(); i++) {
        attachments.push_back(colorAttachment_2);
    }
    attachments.push_back(depthAttachment);
    attachments[0] = colorAttachment;
    VkRenderPassCreateInfo renderPassInfo = vkinit::renderPass_create_info(static_cast<uint32_t>(attachments.size()), attachments.data(), 1, &subpass, 1, &dependency);

    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create render pass!");
    }
}

void RayQueryApp::recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex)
{
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("failed to begin recording command buffer!");
    }
    VkRenderPassBeginInfo renderPassInfo = vkinit::renderPass_begin_info(renderPass, swapChainFramebuffers[imageIndex], swapChainExtent, outPutAttachments.size()+1);

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);   // ray query pass ***************************************************

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    VkBuffer vertexBuffers[] = { vertexBuffer };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

    vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);

    vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

    vkCmdEndRenderPass(commandBuffer);

    {
        VkImageSubresourceLayers subresourceLayers = vkinit::subresource_layers();
        VkOffset3D offset = vkinit::offset();
        VkExtent3D extent = vkinit::extent(WIDTH, HEIGHT);
        VkImageCopy imageCopy = vkinit::imageCopy(subresourceLayers, offset, subresourceLayers, offset, extent);

        //copy 当前帧 渲染结果 到 History(Geometry) Buffer
        vkCmdCopyImage(commandBuffer, swapChainImages[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, inPutAttachments[0].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imageCopy);
        for (auto i = 1; i < outPutAttachments.size(); i++) {
            vkCmdCopyImage(commandBuffer, outPutAttachments[i].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, inPutAttachments[i].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imageCopy);
        }
        vkCmdCopyImage(commandBuffer, depthImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, historyDepth.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imageCopy);
    }

    VkRenderPassBeginInfo renderPassInfo_2th = vkinit::renderPass_begin_info(pipline_filter.renderPass, swapChainFramebuffers[imageIndex], swapChainExtent, outPutAttachments.size() + 1);

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo_2th, VK_SUBPASS_CONTENTS_INLINE);  // 1st filter pass ********************************************************

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipline_filter.graphicsPipeline);

    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

    vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipline_filter.pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);

    vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

    vkCmdEndRenderPass(commandBuffer);

    {
        VkImageSubresourceLayers subresourceLayers = vkinit::subresource_layers();
        VkOffset3D offset = vkinit::offset();
        VkExtent3D extent = vkinit::extent(WIDTH, HEIGHT);
        VkImageCopy imageCopy = vkinit::imageCopy(subresourceLayers, offset, subresourceLayers, offset, extent);

        //copy 当前帧 渲染结果 到 History(Geometry) Buffer
        for (auto i = 1; i < 4; i++) {
            vkCmdCopyImage(commandBuffer, outPutAttachments[i].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, inPutAttachments[i].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imageCopy);
        }
        vkCmdCopyImage(commandBuffer, swapChainImages[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, inPutAttachments[6].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imageCopy);  //outColor to imageVar
    }

    VkRenderPassBeginInfo renderPassInfo_2nd = vkinit::renderPass_begin_info(pipline_filter_2nd.renderPass, swapChainFramebuffers[imageIndex], swapChainExtent, outPutAttachments.size() + 1);

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo_2nd, VK_SUBPASS_CONTENTS_INLINE);  // 2nd filter pass ********************************************************

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipline_filter_2nd.graphicsPipeline);

    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

    vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipline_filter_2nd.pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);

    vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

    vkCmdEndRenderPass(commandBuffer);

    {
        VkImageSubresourceLayers subresourceLayers = vkinit::subresource_layers();
        VkOffset3D offset = vkinit::offset();
        VkExtent3D extent = vkinit::extent(WIDTH, HEIGHT);
        VkImageCopy imageCopy = vkinit::imageCopy(subresourceLayers, offset, subresourceLayers, offset, extent);

        //copy 当前帧 渲染结果 到 History(Geometry) Buffer
        for (auto i = 1; i < 4; i++) {
            vkCmdCopyImage(commandBuffer, outPutAttachments[i].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, inPutAttachments[i].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imageCopy);
        }
    }

    VkRenderPassBeginInfo renderPassInfo_3rd = vkinit::renderPass_begin_info(pipline_filter_3rd.renderPass, swapChainFramebuffers[imageIndex], swapChainExtent, outPutAttachments.size() + 1);

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo_3rd, VK_SUBPASS_CONTENTS_INLINE);  // 3rd filter pass ********************************************************

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipline_filter_3rd.graphicsPipeline);

    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

    vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipline_filter_3rd.pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);

    vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

    vkCmdEndRenderPass(commandBuffer);

    {
        VkImageSubresourceLayers subresourceLayers = vkinit::subresource_layers();
        VkOffset3D offset = vkinit::offset();
        VkExtent3D extent = vkinit::extent(WIDTH, HEIGHT);
        VkImageCopy imageCopy = vkinit::imageCopy(subresourceLayers, offset, subresourceLayers, offset, extent);

        //copy 当前帧 渲染结果 到 History(Geometry) Buffer
        for (auto i = 1; i < 4; i++) {
            vkCmdCopyImage(commandBuffer, outPutAttachments[i].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, inPutAttachments[i].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imageCopy);
        }
    }

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
    }
}

uint64_t RayQueryApp::getBufferDeviceAddress(VkBuffer buffer)
{
    VkBufferDeviceAddressInfoKHR bufferDeviceAI{};
    bufferDeviceAI.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    bufferDeviceAI.buffer = buffer;
    return vkGetBufferDeviceAddressKHR(device, &bufferDeviceAI);
}

void RayQueryApp::createBottomLevelAccelerationStructure()
{
    VkDeviceOrHostAddressConstKHR vertexBufferDeviceAddress{};
    VkDeviceOrHostAddressConstKHR indexBufferDeviceAddress{};

    vertexBufferDeviceAddress.deviceAddress = getBufferDeviceAddress(vertexBuffer);
    indexBufferDeviceAddress.deviceAddress = getBufferDeviceAddress(indexBuffer);

    uint32_t numTriangles = primitives.size();
    uint32_t maxVertex = vertices.size();

    // Build
    VkAccelerationStructureGeometryKHR accelerationStructureGeometry = vks::initializers::accelerationStructureGeometryKHR();
    accelerationStructureGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    accelerationStructureGeometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    accelerationStructureGeometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    accelerationStructureGeometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    accelerationStructureGeometry.geometry.triangles.vertexData = vertexBufferDeviceAddress;
    accelerationStructureGeometry.geometry.triangles.maxVertex = maxVertex;
    accelerationStructureGeometry.geometry.triangles.vertexStride = sizeof(Vertex);
    accelerationStructureGeometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
    accelerationStructureGeometry.geometry.triangles.indexData = indexBufferDeviceAddress;
    accelerationStructureGeometry.geometry.triangles.transformData.deviceAddress = 0;
    accelerationStructureGeometry.geometry.triangles.transformData.hostAddress = nullptr;

    // Get size info
    VkAccelerationStructureBuildGeometryInfoKHR accelerationStructureBuildGeometryInfo = vks::initializers::accelerationStructureBuildGeometryInfoKHR();
    accelerationStructureBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    accelerationStructureBuildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    accelerationStructureBuildGeometryInfo.geometryCount = 1;
    accelerationStructureBuildGeometryInfo.pGeometries = &accelerationStructureGeometry;

    VkAccelerationStructureBuildSizesInfoKHR accelerationStructureBuildSizesInfo = vks::initializers::accelerationStructureBuildSizesInfoKHR();
    vkGetAccelerationStructureBuildSizesKHR(
        device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &accelerationStructureBuildGeometryInfo,
        &numTriangles,
        &accelerationStructureBuildSizesInfo);

    createAccelerationStructure(bottomLevelAS, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, accelerationStructureBuildSizesInfo);

    // Create a small scratch buffer used during build of the bottom level acceleration structure
    ScratchBuffer scratchBuffer = createScratchBuffer(accelerationStructureBuildSizesInfo.buildScratchSize);

    VkAccelerationStructureBuildGeometryInfoKHR accelerationBuildGeometryInfo = vks::initializers::accelerationStructureBuildGeometryInfoKHR();
    accelerationBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    accelerationBuildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    accelerationBuildGeometryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    accelerationBuildGeometryInfo.dstAccelerationStructure = bottomLevelAS.handle;
    accelerationBuildGeometryInfo.geometryCount = 1;
    accelerationBuildGeometryInfo.pGeometries = &accelerationStructureGeometry;
    accelerationBuildGeometryInfo.scratchData.deviceAddress = scratchBuffer.deviceAddress;

    VkAccelerationStructureBuildRangeInfoKHR accelerationStructureBuildRangeInfo{};
    accelerationStructureBuildRangeInfo.primitiveCount = numTriangles;
    accelerationStructureBuildRangeInfo.primitiveOffset = 0;
    accelerationStructureBuildRangeInfo.firstVertex = 0;
    accelerationStructureBuildRangeInfo.transformOffset = 0;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR*> accelerationBuildStructureRangeInfos = { &accelerationStructureBuildRangeInfo };

    // Build the acceleration structure on the device via a one-time command buffer submission
    // Some implementations may support acceleration structure building on the host (VkPhysicalDeviceAccelerationStructureFeaturesKHR->accelerationStructureHostCommands), but we prefer device builds

    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    //VkCommandBuffer commandBuffer = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
    vkCmdBuildAccelerationStructuresKHR(
        commandBuffer,
        1,
        &accelerationBuildGeometryInfo,
        accelerationBuildStructureRangeInfos.data());
    //vulkanDevice->flushCommandBuffer(commandBuffer, queue);
    endSingleTimeCommands(commandBuffer);


    deleteScratchBuffer(scratchBuffer);
}

void RayQueryApp::createTopLevelAccelerationStructure()
{
    VkTransformMatrixKHR transformMatrix = {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f };

    VkAccelerationStructureInstanceKHR instance{};
    instance.transform = transformMatrix;
    instance.instanceCustomIndex = 0;
    instance.mask = 0xFF;
    instance.instanceShaderBindingTableRecordOffset = 0;
    instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    instance.accelerationStructureReference = bottomLevelAS.deviceAddress;

    // Buffer for instance data
    //vks::Buffer instancesBuffer;
    /*
     VK_CHECK_RESULT(createBuffer(
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &instancesBuffer,
        sizeof(VkAccelerationStructureInstanceKHR),
        &instance));
    */
    

    VkBuffer instancesBuffer;
    VkDeviceMemory instancesDeviceMemory;
    createBuffer(sizeof(VkAccelerationStructureInstanceKHR), VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, instancesBuffer, instancesDeviceMemory);
    void* data;
    vkMapMemory(device, instancesDeviceMemory, 0, sizeof(VkAccelerationStructureInstanceKHR), 0, &data);
    memcpy(data, &instance, sizeof(VkAccelerationStructureInstanceKHR));
    vkUnmapMemory(device, instancesDeviceMemory);


    VkDeviceOrHostAddressConstKHR instanceDataDeviceAddress{};
    instanceDataDeviceAddress.deviceAddress = getBufferDeviceAddress(instancesBuffer);

    VkAccelerationStructureGeometryKHR accelerationStructureGeometry = vks::initializers::accelerationStructureGeometryKHR();
    accelerationStructureGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    accelerationStructureGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    accelerationStructureGeometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    accelerationStructureGeometry.geometry.instances.arrayOfPointers = VK_FALSE;
    accelerationStructureGeometry.geometry.instances.data = instanceDataDeviceAddress;

    // Get size info
    VkAccelerationStructureBuildGeometryInfoKHR accelerationStructureBuildGeometryInfo = vks::initializers::accelerationStructureBuildGeometryInfoKHR();
    accelerationStructureBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    accelerationStructureBuildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    accelerationStructureBuildGeometryInfo.geometryCount = 1;
    accelerationStructureBuildGeometryInfo.pGeometries = &accelerationStructureGeometry;

    uint32_t primitive_count = 1;

    VkAccelerationStructureBuildSizesInfoKHR accelerationStructureBuildSizesInfo = vks::initializers::accelerationStructureBuildSizesInfoKHR();
    vkGetAccelerationStructureBuildSizesKHR(
        device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &accelerationStructureBuildGeometryInfo,
        &primitive_count,
        &accelerationStructureBuildSizesInfo);

    createAccelerationStructure(topLevelAS, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, accelerationStructureBuildSizesInfo);

    // Create a small scratch buffer used during build of the top level acceleration structure
    ScratchBuffer scratchBuffer = createScratchBuffer(accelerationStructureBuildSizesInfo.buildScratchSize);

    VkAccelerationStructureBuildGeometryInfoKHR accelerationBuildGeometryInfo = vks::initializers::accelerationStructureBuildGeometryInfoKHR();
    accelerationBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    accelerationBuildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    accelerationBuildGeometryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    accelerationBuildGeometryInfo.dstAccelerationStructure = topLevelAS.handle;
    accelerationBuildGeometryInfo.geometryCount = 1;
    accelerationBuildGeometryInfo.pGeometries = &accelerationStructureGeometry;
    accelerationBuildGeometryInfo.scratchData.deviceAddress = scratchBuffer.deviceAddress;

    VkAccelerationStructureBuildRangeInfoKHR accelerationStructureBuildRangeInfo{};
    accelerationStructureBuildRangeInfo.primitiveCount = 1;
    accelerationStructureBuildRangeInfo.primitiveOffset = 0;
    accelerationStructureBuildRangeInfo.firstVertex = 0;
    accelerationStructureBuildRangeInfo.transformOffset = 0;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR*> accelerationBuildStructureRangeInfos = { &accelerationStructureBuildRangeInfo };

    // Build the acceleration structure on the device via a one-time command buffer submission
    // Some implementations may support acceleration structure building on the host (VkPhysicalDeviceAccelerationStructureFeaturesKHR->accelerationStructureHostCommands), but we prefer device builds
    
    //VkCommandBuffer commandBuffer = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
    /*
        vkCmdBuildAccelerationStructuresKHR(
        commandBuffer,
        1,
        &accelerationBuildGeometryInfo,
        accelerationBuildStructureRangeInfos.data());
    vulkanDevice->flushCommandBuffer(commandBuffer, queue);
    */
    

    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    //VkCommandBuffer commandBuffer = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
    vkCmdBuildAccelerationStructuresKHR(
        commandBuffer,
        1,
        &accelerationBuildGeometryInfo,
        accelerationBuildStructureRangeInfos.data());
    //vulkanDevice->flushCommandBuffer(commandBuffer, queue);
    endSingleTimeCommands(commandBuffer);

    vkDestroyBuffer(device, instancesBuffer, nullptr);
    vkFreeMemory(device, instancesDeviceMemory, nullptr);
}

void RayQueryApp::createAccelerationStructure(AccelerationStructure& accelerationStructure, VkAccelerationStructureTypeKHR type, VkAccelerationStructureBuildSizesInfoKHR buildSizeInfo)
{
    // Buffer and memory
    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = buildSizeInfo.accelerationStructureSize;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    VK_CHECK_RESULT(vkCreateBuffer(device, &bufferCreateInfo, nullptr, &accelerationStructure.buffer));
    VkMemoryRequirements memoryRequirements{};
    vkGetBufferMemoryRequirements(device, accelerationStructure.buffer, &memoryRequirements);
    VkMemoryAllocateFlagsInfo memoryAllocateFlagsInfo{};
    memoryAllocateFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    memoryAllocateFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
    VkMemoryAllocateInfo memoryAllocateInfo{};
    memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocateInfo.pNext = &memoryAllocateFlagsInfo;
    memoryAllocateInfo.allocationSize = memoryRequirements.size;
    memoryAllocateInfo.memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK_RESULT(vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &accelerationStructure.memory));
    VK_CHECK_RESULT(vkBindBufferMemory(device, accelerationStructure.buffer, accelerationStructure.memory, 0));
    // Acceleration structure
    VkAccelerationStructureCreateInfoKHR accelerationStructureCreate_info{};
    accelerationStructureCreate_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    accelerationStructureCreate_info.buffer = accelerationStructure.buffer;
    accelerationStructureCreate_info.size = buildSizeInfo.accelerationStructureSize;
    accelerationStructureCreate_info.type = type;
    vkCreateAccelerationStructureKHR(device, &accelerationStructureCreate_info, nullptr, &accelerationStructure.handle);
    // AS device address
    VkAccelerationStructureDeviceAddressInfoKHR accelerationDeviceAddressInfo{};
    accelerationDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    accelerationDeviceAddressInfo.accelerationStructure = accelerationStructure.handle;
    accelerationStructure.deviceAddress = vkGetAccelerationStructureDeviceAddressKHR(device, &accelerationDeviceAddressInfo);
}

void RayQueryApp::deleteAccelerationStructure(AccelerationStructure& accelerationStructure)
{
    vkFreeMemory(device, accelerationStructure.memory, nullptr);
    vkDestroyBuffer(device, accelerationStructure.buffer, nullptr);
    vkDestroyAccelerationStructureKHR(device, accelerationStructure.handle, nullptr);
}

RayQueryApp::ScratchBuffer RayQueryApp::createScratchBuffer(VkDeviceSize size)
{
    ScratchBuffer scratchBuffer{};
    // Buffer and memory
    VkBufferCreateInfo bufferCreateInfo=vkinit::buffer_create_info(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    VK_CHECK_RESULT(vkCreateBuffer(device, &bufferCreateInfo, nullptr, &scratchBuffer.handle));
    VkMemoryRequirements memoryRequirements{};
    vkGetBufferMemoryRequirements(device, scratchBuffer.handle, &memoryRequirements);
    VkMemoryAllocateFlagsInfo memoryAllocateFlagsInfo{};
    memoryAllocateFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    memoryAllocateFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
    VkMemoryAllocateInfo memoryAllocateInfo = vkinit::memoryAllocate_info(memoryRequirements.size, findMemoryType(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
    memoryAllocateInfo.pNext = &memoryAllocateFlagsInfo;
    VK_CHECK_RESULT(vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &scratchBuffer.memory));
    VK_CHECK_RESULT(vkBindBufferMemory(device, scratchBuffer.handle, scratchBuffer.memory, 0));
    // Buffer device address
    VkBufferDeviceAddressInfoKHR bufferDeviceAddresInfo{};
    bufferDeviceAddresInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    bufferDeviceAddresInfo.buffer = scratchBuffer.handle;
    scratchBuffer.deviceAddress = vkGetBufferDeviceAddressKHR(device, &bufferDeviceAddresInfo);
    return scratchBuffer;
}

void RayQueryApp::deleteScratchBuffer(ScratchBuffer& scratchBuffer)
{
    if (scratchBuffer.memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, scratchBuffer.memory, nullptr);
    }
    if (scratchBuffer.handle != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, scratchBuffer.handle, nullptr);
    }
}
