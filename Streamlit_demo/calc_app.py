import streamlit as st
st.write('# Streamlit Calculator')

#input field for two numbers

number1 = st.number_input('Enter first number')
number2 = st.number_input('Enter second number')


#perform the calculation 

# Perform the calculations
sum_result = number1 + number2
sub_result = number1 - number2
mul_result = number1 * number2
div_result = number1 / number2 if number2 != 0 else "Undefined (cannot divide by zero)"

#display the results 
st.write('#### Results:')
st.write(f'**Addition:** {number1} + {number2} = {sum_result}')
st.write(f'**Subtraction:** {number1} - {number2} = {sub_result}')
st.write(f'**Multiplication:** {number1} * {number2} = {mul_result}')
st.write(f'**Division:** {number1} / {number2} = {div_result}')