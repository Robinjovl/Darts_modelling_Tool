#include "logger.h"

namespace mesh {
logging::Logger &logger = logging::get_logger("discretizer").get_logger("mesh");
}
