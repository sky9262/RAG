from langchain_community.embeddings.bedrock import BedrockEmbeddings
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError

def get_embedding_function():
    try:
        embeddings = BedrockEmbeddings(
            credentials_profile_name="default", region_name="us-east-1"
        )
        return embeddings
    except NoCredentialsError:
        raise ValueError("AWS credentials not found. Please configure your credentials.")
    except PartialCredentialsError:
        raise ValueError("Incomplete AWS credentials. Please check your configuration.")
    except ClientError as e:
        raise ValueError(f"AWS client error: {e}")
    except Exception as e:
        raise ValueError(f"An unexpected error occurred: {e}")
