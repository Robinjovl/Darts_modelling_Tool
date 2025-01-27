#include "logger.h"

namespace dis {
logging::Logger &logger = logging::get_logger("discretizer").get_logger("mesh");
}
