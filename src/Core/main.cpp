#include <Application.h>

#	define LOGI(...)        \
		printf(__VA_ARGS__); \
		printf("\n")
#	define LOGW(...)        \
		printf(__VA_ARGS__); \
		printf("\n")
#	define LOGE(...)                 \
		fprintf(stderr, __VA_ARGS__); \
		fprintf(stderr, "\n")

int main() {

	try
	{
		Application app;
		app.init();
		app.run();
	}
	catch (const std::exception& e)
	{
		LOGE("%s", e.what());
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}