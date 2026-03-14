class DataFetchError(Exception):
    """Raised when NBA API data fetching fails after max retries."""
    pass


class FeatureEngineeringError(Exception):
    """Raised when required fields are missing during feature engineering."""
    pass


class LLMError(Exception):
    """Raised when an LLM API call fails."""
    pass


class ModelNotFoundError(Exception):
    """Raised when a serialized model file cannot be located."""
    pass
