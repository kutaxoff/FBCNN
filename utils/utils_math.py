def calculate_mape(defined_value, estimated_values):
    """
    Calculate the Mean Absolute Percentage Error (MAPE) between a defined number and an array of estimated values.

    Parameters:
    defined_value (float): The defined number.
    estimated_values (list of float): The array of estimated values.

    Returns:
    float: The MAPE value as a percentage.
    """
    # Ensure the defined value is not zero to avoid division by zero
    if defined_value == 0:
        raise ValueError("The defined value must not be zero.")
    
    # Calculate the absolute percentage errors
    absolute_percentage_errors = [abs((defined_value - estimate) / defined_value) * 100 for estimate in estimated_values]
    
    # Calculate the mean of the absolute percentage errors
    mape = sum(absolute_percentage_errors) / len(absolute_percentage_errors)
    
    return mape
