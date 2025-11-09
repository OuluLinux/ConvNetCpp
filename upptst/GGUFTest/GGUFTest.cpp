#include <Core/Core.h>
#include <ConvNet/ConvNet.h>
#include <stdio.h>
#include <string>
#include <vector>

// Include llama.cpp GGUF headers
extern "C" {
    #include "gguf.h"
}

using namespace Upp;

CONSOLE_APP_MAIN
{
    SeedRandom();
    LOG("Starting GGUF file support test with llama.cpp integration... [Force rebuild 2]");

    // Test basic GGUF functionality
    struct gguf_context * gguf_ctx = gguf_init_empty();
    
    if (!gguf_ctx) {
        LOG("ERROR: Failed to initialize GGUF context");
        return;
    }
    
    LOG("✓ GGUF context initialized successfully");

    // Test setting and getting a string value
    const char* test_key = "test_key";
    const char* test_value = "test_value";
    
    gguf_set_val_str(gguf_ctx, test_key, test_value);
    
    int key_id = gguf_find_key(gguf_ctx, test_key);
    if (key_id >= 0) {
        const char* retrieved_value = gguf_get_val_str(gguf_ctx, key_id);
        if (retrieved_value && String(retrieved_value) == String(test_value)) {
            LOG("✓ String value set and retrieved successfully: " << retrieved_value);
        } else {
            LOG("✗ Failed to retrieve string value");
            gguf_free(gguf_ctx);
            return;
        }
    } else {
        LOG("✗ Key not found after setting");
        gguf_free(gguf_ctx);
        return;
    }

    // Test setting and getting a numeric value
    const char* num_key = "number_key";
    uint32_t num_value = 42;
    
    gguf_set_val_u32(gguf_ctx, num_key, num_value);
    
    int num_key_id = gguf_find_key(gguf_ctx, num_key);
    if (num_key_id >= 0) {
        uint32_t retrieved_num = gguf_get_val_u32(gguf_ctx, num_key_id);
        if (retrieved_num == num_value) {
            LOG("✓ Numeric value set and retrieved successfully: " << retrieved_num);
        } else {
            LOG("✗ Failed to retrieve numeric value");
            gguf_free(gguf_ctx);
            return;
        }
    } else {
        LOG("✗ Number key not found after setting");
        gguf_free(gguf_ctx);
        return;
    }

    // Test creating a simple GGUF file using the available API
    bool write_success = gguf_write_to_file(gguf_ctx, "test.gguf", false);  // false = include tensor data

    if (!write_success) {
        LOG("✗ Failed to write complete GGUF file");
        gguf_free(gguf_ctx);
        return;
    }
    
    LOG("✓ GGUF file written successfully");

    // Test reading the GGUF file back
    FILE* read_file = fopen("test.gguf", "rb");
    if (!read_file) {
        LOG("✗ Failed to open test.gguf for reading");
        gguf_free(gguf_ctx);
        return;
    }

    struct gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ nullptr,
    };
    
    struct gguf_context * read_ctx = gguf_init_from_file("test.gguf", params);
    if (!read_ctx) {
        LOG("✗ Failed to read GGUF file");
        fclose(read_file);
        gguf_free(gguf_ctx);
        return;
    }
    
    LOG("✓ GGUF file read successfully");

    // Verify the values in the read context
    int read_key_id = gguf_find_key(read_ctx, test_key);
    if (read_key_id >= 0) {
        const char* read_value = gguf_get_val_str(read_ctx, read_key_id);
        if (read_value && String(read_value) == String(test_value)) {
            LOG("✓ String value verified in read file: " << read_value);
        } else {
            LOG("✗ String value not verified in read file");
        }
    } else {
        LOG("✗ Test key not found in read file");
    }
    
    int read_num_key_id = gguf_find_key(read_ctx, num_key);
    if (read_num_key_id >= 0) {
        uint32_t read_num = gguf_get_val_u32(read_ctx, read_num_key_id);
        if (read_num == num_value) {
            LOG("✓ Numeric value verified in read file: " << read_num);
        } else {
            LOG("✗ Numeric value not verified in read file");
        }
    } else {
        LOG("✗ Number key not found in read file");
    }

    // Clean up
    gguf_free(gguf_ctx);
    gguf_free(read_ctx);
    
    // Clean up test file
    remove("test.gguf");
    
    LOG("GGUF file support test completed successfully!");
}