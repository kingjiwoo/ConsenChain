from setuptools import setup, find_packages

setup(
    name="ConsenChain",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0",
        "langchain-community>=0.0.10",
        "langchain-openai>=0.0.5",
        "langgraph>=0.0.10",
        "openai>=1.0.0",
        "python-dotenv>=1.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "faiss-cpu>=1.7.4",
        "chromadb>=0.4.0",
        "web3>=6.0.0",
        "fpdf>=1.7.2",
        "pydantic>=2.0.0",
    ],
    python_requires=">=3.9",
) 