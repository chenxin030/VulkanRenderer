#pragma once

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include "Camera.h"
#include "Platform.h"
#include "ResourceManager.h"

#include <optional>
#include <string>
#include <vector>

struct Scene;

inline const std::vector<char const*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
inline constexpr bool enableValidationLayers = false;
#else
inline constexpr bool enableValidationLayers = true;
#endif

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    std::optional<uint32_t> computeFamily;
    std::optional<uint32_t> transferFamily;

    [[nodiscard]] bool isComplete() const
    {
        return graphicsFamily.has_value() && presentFamily.has_value() && computeFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
};

struct VulkanBase {
    VulkanBase();
    virtual ~VulkanBase() = default;

    const uint32_t MAX_FRAMES_IN_FLIGHT = 2u;
    uint32_t currentFrame = 0;

    vk::raii::Context context;
    vk::raii::Instance instance = nullptr;
    vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;

    vk::raii::PhysicalDevice physicalDevice = nullptr;
    vk::raii::Device device = nullptr;

    QueueFamilyIndices queueFamilyIndices;
    vk::raii::Queue graphicsQueue = nullptr;
    vk::raii::Queue presentQueue = nullptr;
    vk::raii::Queue computeQueue = nullptr;
    vk::raii::Queue transferQueue = nullptr;

    vk::raii::SurfaceKHR surface = nullptr;

    const std::vector<const char*> requiredDeviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
    std::vector<const char*> deviceExtensions;

    vk::raii::SwapchainKHR swapChain = nullptr;
    std::vector<vk::Image> swapChainImages;
    vk::Format swapChainImageFormat = vk::Format::eUndefined;
    vk::Extent2D swapChainExtent;
    std::vector<vk::raii::ImageView> swapChainImageViews;
    std::vector<vk::ImageLayout> swapChainImageLayouts;

    vk::raii::CommandPool commandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> commandBuffers;

    std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
    std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
    std::vector<vk::raii::Fence> inFlightFences;

    bool framebufferResized = false;

    TextureData depthData;
    vk::ImageLayout depthImageLayout = vk::ImageLayout::eUndefined;

    Camera camera = Camera(glm::vec3(0.0f, 0.0f, 5.0f));
    Platform* platform = nullptr;

    void initialize(Platform* _platform);
    void processInput(float deltaTime) {
        if (!platform->rightMouseButtonPressed) return;
        if (glfwGetKey(platform->window, GLFW_KEY_W) == GLFW_PRESS) camera.ProcessKeyboard(FORWARD, deltaTime);
        if (glfwGetKey(platform->window, GLFW_KEY_S) == GLFW_PRESS) camera.ProcessKeyboard(BACKWARD, deltaTime);
        if (glfwGetKey(platform->window, GLFW_KEY_A) == GLFW_PRESS) camera.ProcessKeyboard(LEFT, deltaTime);
        if (glfwGetKey(platform->window, GLFW_KEY_D) == GLFW_PRESS) camera.ProcessKeyboard(RIGHT, deltaTime);
        if (glfwGetKey(platform->window, GLFW_KEY_Q) == GLFW_PRESS) camera.ProcessKeyboard(UP, deltaTime);
        if (glfwGetKey(platform->window, GLFW_KEY_E) == GLFW_PRESS) camera.ProcessKeyboard(DOWN, deltaTime);
    }

    // Core lifecycle (implemented today in Renderer_*; will be moved later)
    bool initVulkan(const std::string& appName);
    bool createInstance(const std::string& appName);
    bool setupDebugMessenger();
    bool createSurface();
    bool pickPhysicalDevice();
    bool createLogicalDevice();
    bool createSwapChain();
    void cleanupSwapChain();
    virtual void recreateSwapChain();
    bool createImageViews();
    bool createCommandPool();
    bool createCommandBuffers();
    bool createSyncObjects();
    bool createDepthResources();

    bool checkValidationLayerSupport() const;
    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const;
    std::vector<char> readFile(const std::string& filename);
    vk::raii::ShaderModule createShaderModule(const std::vector<char>& code);

    QueueFamilyIndices findQueueFamilies(const vk::raii::PhysicalDevice& device);
    SwapChainSupportDetails querySwapChainSupport(const vk::raii::PhysicalDevice& device);
    static uint32_t chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const& surfaceCapabilities);
    bool checkDeviceExtensionSupport(vk::raii::PhysicalDevice& device);
    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats);
    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes);
    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities);

    vk::Format findSupportedFormat(const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features);
    vk::Format findDepthFormat();
    bool hasStencilComponent(vk::Format format);

    void transition_image_layout(
        vk::Image image,
        vk::ImageLayout old_layout,
        vk::ImageLayout new_layout,
        vk::AccessFlags2 src_access_mask,
        vk::AccessFlags2 dst_access_mask,
        vk::PipelineStageFlags2 src_stage_mask,
        vk::PipelineStageFlags2 dst_stage_mask,
        vk::ImageAspectFlags image_aspect_flags);

    void createVertexBuffer(Mesh& mesh);
    void createIndexBuffer(Mesh& mesh);
    void createUniformBuffers(MeshBuffer& meshResource, vk::DeviceSize size);
    void createUniformBuffers(MeshBuffer& meshResource, vk::DeviceSize size, uint32_t count);
    void createStorageBuffers(MeshBuffer& meshResource, vk::DeviceSize size);
    void createStorageBuffers(MeshBuffer& meshResource, vk::DeviceSize size, vk::BufferUsageFlags usage);
    void createStorageBuffers(MeshBuffer& meshResource, vk::DeviceSize size, vk::BufferUsageFlags usage, uint32_t count);
    void createBuffer(
        vk::DeviceSize size,
        vk::BufferUsageFlags usage,
        vk::MemoryPropertyFlags properties,
        vk::raii::Buffer& buffer, vk::raii::DeviceMemory& bufferMemory);
    void copyBuffer(vk::raii::Buffer& srcBuffer, vk::raii::Buffer& dstBuffer, vk::DeviceSize size);
    void LoadHDRTextureFromFile(const std::string& path, TextureData& texData);
    void LoadTextureFromFile(const std::string& path, TextureData& texData);
    void generateMipmaps(vk::raii::Image& image, vk::Format imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels);
    void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, TextureData& texData);
    void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, uint32_t arrayLayers, vk::ImageCreateFlags flags, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, TextureData& texData);
    vk::raii::ImageView createImageView(vk::raii::Image& image, vk::Format format, vk::ImageAspectFlags aspectFlags, uint32_t mipLevels);
    void createTextureSampler(vk::raii::Sampler& textureSampler);
    void transitionImageLayout(const vk::raii::Image& image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout, uint32_t mipLevels);
    void copyBufferToImage(const vk::raii::Buffer& buffer, vk::raii::Image& image, uint32_t width, uint32_t height);
    std::unique_ptr<vk::raii::CommandBuffer> beginSingleTimeCommands();
    void endSingleTimeCommands(vk::raii::CommandBuffer& commandBuffer);

    // ImGui UI
    bool uiEnabled = true;
    vk::raii::DescriptorSetLayout uiDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool uiDescriptorPool = nullptr;
    vk::raii::PipelineLayout uiPipelineLayout = nullptr;
    vk::raii::Pipeline uiPipeline = nullptr;
    vk::raii::DescriptorSets uiDescriptorSets = nullptr;
    TextureData uiFontTexture;

    struct UiFrameBuffers
    {
        vk::raii::Buffer vertexBuffer = nullptr;
        vk::raii::DeviceMemory vertexBufferMemory = nullptr;
        void* vertexMapped = nullptr;
        size_t vertexSize = 0;
        size_t vertexBufferSize = 0; // tracked for dynamic reallocation
        vk::raii::Buffer indexBuffer = nullptr;
        vk::raii::DeviceMemory indexBufferMemory = nullptr;
        void* indexMapped = nullptr;
        size_t indexSize = 0;
        size_t indexBufferSize = 0; // tracked for dynamic reallocation
    };
    std::vector<UiFrameBuffers> uiFrameBuffers;

    virtual void shutdownUI();
};

