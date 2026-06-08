def mock_ml_inference(data_string):
    """
    Dummy ML Logic: Predicts '1' if the message contains 'alert', else '0'.
    """
    if not data_string:
        return 0
    return 1 if "alert" in data_string.lower() else 0
