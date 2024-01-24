from generic_logger import logger


def main():
    logger.critical("Something bad happened.", extra={
        "user": "johndoe", "session_id": "123456"})


if __name__ == '__main__':
    main()
