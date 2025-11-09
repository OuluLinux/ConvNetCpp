#include "PerfBenchmarkTest.h"

using namespace Upp;
using namespace ConvNet;

namespace PerfBenchmarkTest {
	
void Main() {
    printf("ConvNetCpp Performance Testing Framework Demo\n");
    printf("=============================================\n");
    
    // Run the comprehensive benchmark suite
    RunPerformanceBenchmarks();
    
    printf("Performance testing completed.\n");
}

}