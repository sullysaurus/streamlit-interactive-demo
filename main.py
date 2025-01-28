import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt

def display_tensor_section():
    st.title("Introduction to Tensors")
    
    # Real-world Applications
    st.write("""
    ## Real-world Applications of Tensors
    
    Tensors are not just mathematical concepts - they're powerful tools used in many real-world applications:
    
    1. **Stock Market Analysis** 📈
       - Stock prices over time form a 2D tensor (time × price)
       - Multiple stocks create a 3D tensor (time × stock × price)
       - Technical indicators add a 4th dimension (time × stock × indicator × value)
    
    2. **Face Recognition** 👤
       - Images are 3D tensors (height × width × color channels)
       - Facial features are encoded as tensors
       - Neural networks process these tensors to identify faces
    
    3. **Natural Language Processing** 📝
       - Words are converted to tensors (word embeddings)
       - Sentences become sequences of tensors
       - Language models use these tensors to understand context
    """)
    
    # Interactive Tensor Creation
    st.subheader("Create Your Own Tensor")
    
    # 1D Tensor Creation
    st.write("""
    ### 1D Tensor Creator
    Think of a 1D tensor like a row in a spreadsheet or a list of stock prices for one day.
    Try creating your own!
    """)
    
    num_elements = st.slider("Number of elements", 1, 6, 4)
    values = []
    cols = st.columns(num_elements)
    for i in range(num_elements):
        with cols[i]:
            values.append(st.number_input(f"Value {i+1}", value=float(i+1), format="%.2f"))
    
    tensor_1d = torch.tensor(values)
    st.write("Your 1D Tensor:")
    st.write(tensor_1d)
    
    # 2D Tensor Creation
    st.write("""
    ### 2D Tensor Creator
    A 2D tensor is like a table or matrix. In image processing, it could represent a grayscale image.
    In finance, it might represent stock prices for multiple companies over time.
    """)
    
    rows = st.slider("Number of rows", 1, 4, 2)
    cols = st.slider("Number of columns", 1, 4, 2)
    
    values_2d = []
    for i in range(rows):
        row_values = []
        row_cols = st.columns(cols)
        for j in range(cols):
            with row_cols[j]:
                row_values.append(st.number_input(f"Value [{i},{j}]", value=float(i*cols + j + 1), format="%.2f"))
        values_2d.append(row_values)
    
    tensor_2d = torch.tensor(values_2d)
    st.write("Your 2D Tensor:")
    st.write(tensor_2d)
    
    # Interactive Operations
    st.subheader("Tensor Operations Playground")
    st.write("""
    Just like in real applications, we can perform various operations on tensors.
    For example, in image processing, we might scale pixel values or in finance,
    we might normalize stock prices.
    """)
    
    operation = st.selectbox("Select an operation", 
                            ["Add a scalar", "Multiply by a scalar", "Calculate mean", "Calculate sum"])
    
    if operation == "Add a scalar":
        scalar = st.number_input("Enter a number to add", value=1.0, format="%.2f")
        st.write(f"Result of adding {scalar}:")
        st.write(tensor_1d + scalar)
        st.write(tensor_2d + scalar)
    
    elif operation == "Multiply by a scalar":
        scalar = st.number_input("Enter a number to multiply", value=2.0, format="%.2f")
        st.write(f"Result of multiplying by {scalar}:")
        st.write(tensor_1d * scalar)
        st.write(tensor_2d * scalar)
    
    elif operation == "Calculate mean":
        st.write("Mean of 1D tensor:", tensor_1d.mean().item())
        st.write("Mean of 2D tensor:", tensor_2d.mean().item())
    
    elif operation == "Calculate sum":
        st.write("Sum of 1D tensor:", tensor_1d.sum().item())
        st.write("Sum of 2D tensor:", tensor_2d.sum().item())
    
    # Tensor Challenge
    st.subheader("🎯 Tensor Challenge")
    st.write("""
    Let's solve a real-world inspired challenge! Imagine you're processing financial data
    and need to scale some values to match a target sum.
    """)
    
    # Generate random tensors for the challenge
    challenge_tensor = torch.tensor([2.0, 4.0, 6.0, 8.0])
    st.write("Given tensor (representing financial values):", challenge_tensor)
    
    target_sum = 40.0
    st.write("""
    Your goal is to find the scaling factor that will make the sum of all values equal to 40.0.
    This is similar to normalizing financial data or scaling features in machine learning.
    
    💡 Hint: Think about the relationship between the current sum and the target sum!
    """)
    
    user_operation = st.number_input("Enter a scalar to multiply the tensor with to get a sum of 40.0", value=1.0, format="%.2f")
    
    if st.button("Check Answer", key="tensor_challenge"):
        try:
            scalar = float(user_operation)
            result = challenge_tensor * scalar
            current_sum = result.sum().item()
            st.write("Your result:", result)
            st.write(f"Sum of your result: {current_sum}")
            
            if abs(current_sum - target_sum) < 0.01:
                st.success("🎉 Congratulations! You solved the challenge!")
            else:
                st.error("Not quite right. Try again!")
                if current_sum < target_sum:
                    st.write("Hint: Try a larger number")
                else:
                    st.write("Hint: Try a smaller number")
        except ValueError:
            st.error("Please enter a valid number")
    
    if st.button("Show Solution", key="tensor_solution"):
        st.write("""
        ### Solution Explanation
        
        To solve this challenge, we need to find a scalar that when multiplied by our tensor
        will give us a sum of 40.0. Here's how to think about it:
        
        1. Current tensor sum = 2 + 4 + 6 + 8 = 20
        2. Target sum = 40
        3. Scaling factor = Target sum ÷ Current sum = 40 ÷ 20 = 2
        
        Therefore, multiplying the tensor by 2 will give us the desired result:
        [2, 4, 6, 8] × 2 = [4, 8, 12, 16]
        
        This is exactly how data scientists scale their data in real applications:
        - In finance: normalizing stock prices across different scales
        - In machine learning: scaling features to a specific range
        - In image processing: normalizing pixel values
        """)

def display_operations_section():
    st.title("Tensor Operations")
    
    # Introduction
    st.write("""
    Welcome to the Tensor Operations section! Here, you'll learn how to manipulate and transform
    tensors using PyTorch's powerful operations. These operations are essential for:
    
    - Data preprocessing in machine learning
    - Image transformation in computer vision
    - Mathematical computations in scientific computing
    """)
    
    # Basic Operations Section
    st.header("Basic Tensor Operations")
    st.write("""
    Let's explore fundamental tensor operations that are commonly used in real-world applications.
    We'll start with creating tensors and performing basic manipulations.
    """)
    
    # Create sample tensors
    st.subheader("Sample Tensors")
    st.write("Let's create some sample tensors to work with:")
    
    # Create tensors with user input
    cols = st.columns(2)
    with cols[0]:
        tensor_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        st.write("Tensor A:")
        st.write(tensor_a)
    
    with cols[1]:
        tensor_b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        st.write("Tensor B:")
        st.write(tensor_b)
    
    # Operations Playground
    st.subheader("Operations Playground")
    operation = st.selectbox(
        "Select an operation to perform:",
        ["Addition", "Subtraction", "Matrix Multiplication", "Element-wise Multiplication"]
    )
    
    result = None
    if operation == "Addition":
        result = tensor_a + tensor_b
        explanation = """
        Addition combines corresponding elements from both tensors.
        This is commonly used in:
        - Combining feature vectors in machine learning
        - Updating neural network weights during training
        - Merging multiple data sources
        """
    elif operation == "Subtraction":
        result = tensor_a - tensor_b
        explanation = """
        Subtraction finds the difference between corresponding elements.
        Applications include:
        - Computing gradients in neural networks
        - Finding differences between feature vectors
        - Error calculation in machine learning models
        """
    elif operation == "Matrix Multiplication":
        result = torch.matmul(tensor_a, tensor_b)
        explanation = """
        Matrix multiplication is fundamental in:
        - Neural network layer computations
        - Image transformations
        - Feature extraction in deep learning
        """
    elif operation == "Element-wise Multiplication":
        result = tensor_a * tensor_b
        explanation = """
        Element-wise multiplication (Hadamard product) is used in:
        - Applying masks or filters
        - Feature scaling
        - Attention mechanisms in neural networks
        """
    
    if result is not None:
        st.write("Result:")
        st.write(result)
        st.write("### Understanding the Operation")
        st.write(explanation)
    
    # Interactive Challenge
    st.subheader("🎯 Operation Challenge")
    st.write("""
    Let's test your understanding! Given two tensors C and D below,
    try to predict the result of different operations before revealing the answer.
    """)
    
    tensor_c = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    tensor_d = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    
    st.write("Tensor C:")
    st.write(tensor_c)
    st.write("Tensor D:")
    st.write(tensor_d)
    
    challenge_operation = st.selectbox(
        "Select an operation to try:",
        ["Matrix Multiplication", "Element-wise Multiplication"],
        key="challenge_op"
    )
    
    # Calculate correct results first
    matrix_mult_result = torch.matmul(tensor_c, tensor_d).numpy().tolist()
    element_mult_result = (tensor_c * tensor_d).numpy().tolist()
    
    # Create dropdown options based on operation
    if challenge_operation == "Matrix Multiplication":
        options = [
            "[[2, 2], [2, 2]]",  # Correct answer
            "[[2, 0], [0, 2]]",
            "[[1, 1], [1, 1]]",
            "[[4, 4], [4, 4]]"
        ]
        correct_result = "[[2, 2], [2, 2]]"
    else:  # Element-wise Multiplication
        options = [
            "[[2, 0], [0, 2]]",  # Correct answer
            "[[2, 2], [2, 2]]",
            "[[1, 1], [1, 1]]",
            "[[4, 0], [0, 4]]"
        ]
        correct_result = "[[2, 0], [0, 2]]"
    
    user_prediction = st.selectbox(
        "Select your prediction for the result:",
        options,
        key="prediction_dropdown"
    )
    
    if st.button("Check Answer", key="operations_challenge"):
        st.write("Your prediction:", user_prediction)
        st.write("Correct answer:", correct_result)
        if user_prediction == correct_result:
            st.success("🎉 Correct! You've understood the operation.")
        else:
            st.error("Not quite right. Try again!")
            if challenge_operation == "Matrix Multiplication":
                st.write("Hint: Think about how matrix multiplication combines rows and columns!")
            else:
                st.write("Hint: Remember that element-wise multiplication multiplies corresponding elements!")
    
    if st.button("Show Solution", key="operations_solution"):
        if challenge_operation == "Matrix Multiplication":
            st.write("""
            ### Matrix Multiplication Solution
            
            When multiplying these matrices:
            ```
            C = [[2, 0],   D = [[1, 1],
                [0, 2]]        [1, 1]]
            
            The result is:
            ```
            [[2, 2],
             [2, 2]]
            ```
            
            This happens because:
            1. Multiplies rows of C with columns of D
            2. The diagonal matrix C acts as a scaling matrix
            3. Result shows how matrix multiplication can be used for scaling and transformation
            
            Common applications:
            - Neural networks transform input features
            - Computer vision applies transformations to images
            - Natural language models process word embeddings
            """)
        else:
            st.write("""
            ### Element-wise Multiplication Solution
            
            When multiplying these matrices element by element:
            ```
            C = [[2, 0],   D = [[1, 1],
                [0, 2]]        [1, 1]]
            ```
            
            The result is:
            ```
            [[2, 0],
             [0, 2]]
            ```
            
            This happens because:
            1. Multiplies corresponding elements
            2. The diagonal matrix C acts as a mask
            3. Shows how element-wise operations can selectively scale features
            
            Common applications:
            - Applying attention masks in transformers
            - Feature selection in deep learning
            - Image masking in computer vision
            """)

def display_neural_networks_section():
    st.title("Neural Network Basics")
    
    # Introduction
    st.write("""
    Welcome to Neural Networks! This section will introduce you to the building blocks of deep learning.
    Neural networks are powerful tools that can learn patterns from data and make predictions.
    
    Think of a neural network like a digital brain:
    - It has neurons (nodes) that process information
    - These neurons are connected in layers
    - Information flows from input to output through these layers
    """)
    
    # Basic Structure Visualization
    st.subheader("Neural Network Structure")
    
    # Create a simple visualization of a neural network
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define layer sizes
    input_size = 3
    hidden_size = 4
    output_size = 2
    
    # Node positions
    layer_positions = [0.2, 0.5, 0.8]
    node_positions = {
        'input': [(0.2, i) for i in np.linspace(0.2, 0.8, input_size)],
        'hidden': [(0.5, i) for i in np.linspace(0.2, 0.8, hidden_size)],
        'output': [(0.8, i) for i in np.linspace(0.3, 0.7, output_size)]
    }
    
    # Draw connections
    for i_pos in node_positions['input']:
        for h_pos in node_positions['hidden']:
            ax.plot([i_pos[0], h_pos[0]], [i_pos[1], h_pos[1]], 'gray', alpha=0.5)
    
    for h_pos in node_positions['hidden']:
        for o_pos in node_positions['output']:
            ax.plot([h_pos[0], o_pos[0]], [h_pos[1], o_pos[1]], 'gray', alpha=0.5)
    
    # Draw nodes
    for layer, positions in node_positions.items():
        for pos in positions:
            ax.add_patch(plt.Circle(pos, 0.03, color='blue' if layer == 'input' else 'green' if layer == 'hidden' else 'red'))
    
    # Customize plot
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Add labels
    plt.text(0.2, 0.95, 'Input Layer', horizontalalignment='center')
    plt.text(0.5, 0.95, 'Hidden Layer', horizontalalignment='center')
    plt.text(0.8, 0.95, 'Output Layer', horizontalalignment='center')
    
    st.pyplot(fig)
    
    st.write("""
    This diagram shows a simple neural network with:
    - Input Layer (blue): Receives the initial data
    - Hidden Layer (green): Processes the information
    - Output Layer (red): Produces the final prediction
    
    The lines represent connections (weights) between neurons. During training,
    these weights are adjusted to improve predictions.
    """)
    
    # Interactive Example
    st.subheader("Interactive Neural Network Example")
    st.write("""
    Let's explore how a simple neural network processes inputs to make predictions.
    In this example, we'll create a tiny network that tries to learn a simple pattern.
    """)
    
    # Create a simple neural network
    input_value = st.slider("Input Value", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    
    # Simple forward pass simulation
    weight1 = 0.5
    weight2 = 0.8
    bias = 1.0
    
    hidden_value = input_value * weight1 + bias
    output_value = hidden_value * weight2
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Input", f"{input_value:.2f}")
    with col2:
        st.metric("Hidden Layer", f"{hidden_value:.2f}")
    with col3:
        st.metric("Output", f"{output_value:.2f}")
    
    st.write("""
    This simple example shows how:
    1. The input value is received
    2. It's transformed by the hidden layer (using weights and bias)
    3. The final output is produced
    
    In real applications, neural networks:
    - Have many more neurons and layers
    - Learn complex patterns from data
    - Can solve problems like:
        - Image recognition 🖼️
        - Language translation 💬
        - Game playing 🎮
        - And much more!
    """)
    
    # Challenge Section
    st.subheader("🎯 Neural Network Challenge")
    st.write("""
    Can you predict what the output will be for different input values?
    Try to understand the pattern:
    - Input → Hidden Layer: multiplied by 0.5 and add 1.0
    - Hidden Layer → Output: multiplied by 0.8
    """)
    
    user_guess = st.number_input("What do you think the output will be for input = 8.0?", 
                                 min_value=0.0, max_value=20.0, value=5.0, step=0.1)
    
    if st.button("Check Answer", key="nn_challenge"):
        actual_output = (8.0 * 0.5 + 1.0) * 0.8
        if abs(user_guess - actual_output) < 0.1:
            st.success("🎉 Correct! You've understood how the values flow through the network.")
        else:
            st.error(f"Not quite right. The actual output is {actual_output:.2f}.")
    
    if st.button("Show Solution", key="nn_solution"):
        st.write("""
        ### Solution Explanation
        
        Let's break down how the neural network processes the input value of 8.0:
        
        1. Input Layer → Hidden Layer:
           - Input value = 8.0
           - Multiply by weight1 (0.5): 8.0 × 0.5 = 4.0
           - Add bias (1.0): 4.0 + 1.0 = 5.0
           - Hidden layer value = 5.0
        
        2. Hidden Layer → Output Layer:
           - Hidden value = 5.0
           - Multiply by weight2 (0.8): 5.0 × 0.8 = 4.0
           - Final output = 4.0
        
        This simple example demonstrates the basic principles of neural networks:
        - Information flows through layers
        - Each connection has a weight
        - Biases are added to introduce non-linearity
        - The final output is a result of all these transformations
        
        In real neural networks:
        - There are many more neurons and connections
        - Weights and biases are learned from data
        - Non-linear activation functions add complexity
        - The network can learn to approximate any function
        """)

def main():
    st.title("PyTorch for Beginners: A Comprehensive Learning Path")
    
    # Course Overview
    with st.expander("📚 Course Overview"):
        st.write("""
        Welcome to PyTorch for Beginners! This comprehensive course will guide you through the fundamentals
        of PyTorch, one of the most popular deep learning frameworks used in industry and research.
        
        PyTorch is used by companies like Meta, Microsoft, and Tesla for:
        - Building AI models for image and speech recognition
        - Developing natural language processing systems
        - Creating recommendation engines
        - Powering autonomous vehicles
        """)
        
        # Course Prerequisites and Learning Objectives
        st.write("""
        Prerequisites:
        -----------
        1. Basic Python Knowledge
           - Understanding of variables, functions, and loops
           - Familiarity with Python data structures (lists, dictionaries)
           - Basic experience with Python libraries (NumPy is a plus)
        
        2. Basic Mathematics
           - High school level algebra
           - Basic understanding of matrices and vectors
           - Familiarity with simple statistics (mean, standard deviation)
        
        Learning Objectives:
        -----------------
        By the end of this course, you will be able to:
        
        1. Understand PyTorch Fundamentals
           - Work with tensors and understand their importance
           - Perform basic tensor operations
           - Use PyTorch's built-in functions effectively
        
        2. Build Neural Networks
           - Understand neural network architecture
           - Create simple neural networks using PyTorch
           - Train and evaluate models
        
        3. Apply PyTorch to Real Problems
           - Implement practical machine learning solutions
           - Understand common use cases and applications
           - Debug and optimize PyTorch code
        """)
    
    # Display sections in expanders
    with st.expander("🔢 Tensor Basics"):
        display_tensor_section()
    with st.expander("⚡ Tensor Operations"):
        display_operations_section()
    with st.expander("🧠 Neural Networks"):
        display_neural_networks_section()

if __name__ == "__main__":
    main()