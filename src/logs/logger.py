#%%
import logging
import colorlog
import os
import yaml
from logs.constants import LOG_COLORS
#%%

class Logger:
    def __init__(self, name, settings):
        # config
        self.settings = settings
        self.logger = logging.getLogger(name)

    def basicConfig(self):
        """
        Set up the basic configuration for the logger.
        """
        if not self.logger.hasHandlers():
            self.logger.setLevel(logging.DEBUG)

            formatter = colorlog.ColoredFormatter(
                '%(log_color)s[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
                log_colors=LOG_COLORS
            )

            color_handler = colorlog.StreamHandler()
            color_handler.setFormatter(formatter)
            self.logger.addHandler(color_handler)

            # Archivo opcional
            if self.settings["SAVE_LOGS"]:
                os.makedirs(os.path.dirname(self.settings["LOGS_FILE_PATH"]), exist_ok=True)
                file_handler = logging.FileHandler(self.settings["LOGS_FILE_PATH"])
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
    
    def get(self):
        """
        Get the logger instance.
        """
        self.basicConfig()
        return self.logger

if __name__ == "__main__":
    logger = Logger(name="prueba", config_path="configs/base/global_config.yaml").get()
    logger.info("Prueba info")
    logger.debug("Prueba debug")
    logger.warning("Prueba warning")
