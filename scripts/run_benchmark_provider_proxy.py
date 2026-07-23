import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "qmshe.benchmark_framework.provider_proxy:app",
        host="127.0.0.1",
        port=18190,
        log_level="warning",
    )
