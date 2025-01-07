import tkinter as tk
from tkinter import ttk
import string

# Caesar Cipher Encryption
def caesar_cipher_encrypt(text, shift):
    encrypted_text = ''
    for char in text:
        if char.isalpha():
            shifted = ord(char) + shift
            if char.islower():
                shifted = (shifted - ord('a')) % 26 + ord('a')
            elif char.isupper():
                shifted = (shifted - ord('A')) % 26 + ord('A')
            encrypted_text += chr(shifted)
        else:
            encrypted_text += char
    return encrypted_text

# Caesar Cipher Decryption
def caesar_cipher_decrypt(ciphertext, shift):
    decrypted_text = ''
    for char in ciphertext:
        if char.isalpha():
            shifted = ord(char) - shift
            if char.islower():
                shifted = (shifted - ord('a')) % 26 + ord('a')
            elif char.isupper():
                shifted = (shifted - ord('A')) % 26 + ord('A')
            decrypted_text += chr(shifted)
        else:
            decrypted_text += char
    return decrypted_text

# Mono Alphabetic Cipher Encryption
def monoalphabetic_encrypt(plaintext, key):
    key = key.upper()
    alphabet = string.ascii_uppercase
    cipher_dict = {alphabet[i]: key[i] for i in range(len(alphabet))}
    
    encrypted_text = ''
    for char in plaintext.upper():
        if char.isalpha():
            encrypted_text += cipher_dict[char]
        else:
            encrypted_text += char
    return encrypted_text

# Mono Alphabetic Cipher Decryption
def monoalphabetic_decrypt(ciphertext, key):
    key = key.upper()
    alphabet = string.ascii_uppercase
    decipher_dict = {key[i]: alphabet[i] for i in range(len(alphabet))}
    
    decrypted_text = ''
    for char in ciphertext.upper():
        if char.isalpha():
            decrypted_text += decipher_dict[char]
        else:
            decrypted_text += char
    return decrypted_text
# POlY ALPHABETIC ENCRYPTION
def vigenere_cipher_encrypt(plaintext, key):
    plaintext = plaintext.upper()
    key = key.upper()
    key_repeated = (key * (len(plaintext) // len(key))) + key[:len(plaintext) % len(key)]
    result = ''

    for i in range(len(plaintext)):
        char = plaintext[i]
        if char.isalpha():
            shift = ord(key_repeated[i]) - 65
            shifted_char = chr(((ord(char) - 65 + shift) % 26) + 65)
            result += shifted_char
        else:
            result += char

    return result
#POLY ALPHABETIC DECRYPTION
def vigenere_cipher_decrypt(ciphertext, key):
    ciphertext = ciphertext.upper()
    key = key.upper()
    key_repeated = (key * (len(ciphertext) // len(key))) + key[:len(ciphertext) % len(key)]
    result = ''

    for i in range(len(ciphertext)):
        char = ciphertext[i]
        if char.isalpha():
            shift = ord(key_repeated[i]) - 65
            shifted_char = chr(((ord(char) - 65 - shift) % 26) + 65)
            result += shifted_char
        else:
            result += char

    return result
#PLAY FAIR ENCRYPTION
def prepare_input(text):
    text = text.upper().replace(" ", "").replace("J", "I")  # Convert to uppercase and replace 'J' with 'I'
    text_pairs = [text[i:i + 2] for i in range(0, len(text), 2)]  # Split text into pairs of two characters
    return text_pairs


def create_playfair_matrix(key):
    alphabet = "ABCDEFGHIKLMNOPQRSTUVWXYZ"  # 'J' is excluded
    key = key.upper().replace("J", "I")
    key = "".join(dict.fromkeys(key))  # Remove duplicates preserving order

    matrix = []
    for char in key:
        if char not in matrix:
            matrix.append(char)

    for char in alphabet:
        if char not in matrix:
            matrix.append(char)

    playfair_matrix = [matrix[i:i + 5] for i in range(0, 25, 5)]
    return playfair_matrix


def encrypt_playfair(plaintext, key):
    ciphertext_pairs = prepare_input(plaintext)
    matrix = create_playfair_matrix(key)
    encrypted_text = ""

    for pair in ciphertext_pairs:
        char1, char2 = pair[0], pair[1]
        row1, col1, row2, col2 = 0, 0, 0, 0

        # Find positions of characters in the matrix
        for i in range(5):
            for j in range(5):
                if matrix[i][j] == char1:
                    row1, col1 = i, j
                if matrix[i][j] == char2:
                    row2, col2 = i, j

        if row1 == row2:  # Same row
            encrypted_text += matrix[row1][(col1 + 1) % 5] + matrix[row2][(col2 + 1) % 5]
        elif col1 == col2:  # Same column
            encrypted_text += matrix[(row1 + 1) % 5][col1] + matrix[(row2 + 1) % 5][col2]
        else:  # Different row and column
            encrypted_text += matrix[row1][col2] + matrix[row2][col1]

    return encrypted_text
#play fair decryption 
def prepare_input(text):
    text = text.upper().replace(" ", "").replace("J", "I")  # Convert to uppercase and replace 'J' with 'I'
    text_pairs = [text[i:i + 2] for i in range(0, len(text), 2)]  # Split text into pairs of two characters
    return text_pairs


def create_playfair_matrix(key):
    alphabet = "ABCDEFGHIKLMNOPQRSTUVWXYZ"  # 'J' is excluded
    key = key.upper().replace("J", "I")
    key = "".join(dict.fromkeys(key))  # Remove duplicates preserving order

    matrix = []
    for char in key:
        if char not in matrix:
            matrix.append(char)

    for char in alphabet:
        if char not in matrix:
            matrix.append(char)

    playfair_matrix = [matrix[i:i + 5] for i in range(0, 25, 5)]
    return playfair_matrix


def find_position(matrix, char):
    for i in range(5):
        for j in range(5):
            if matrix[i][j] == char:
                return i, j


def decrypt_playfair(ciphertext, key):
    ciphertext_pairs = prepare_input(ciphertext)
    matrix = create_playfair_matrix(key)
    decrypted_text = ""

    for pair in ciphertext_pairs:
        row1, col1 = find_position(matrix, pair[0])
        row2, col2 = find_position(matrix, pair[1])

        if row1 == row2:  # Same row
            decrypted_text += matrix[row1][(col1 - 1) % 5] + matrix[row2][(col2 - 1) % 5]
        elif col1 == col2:  # Same column
            decrypted_text += matrix[(row1 - 1) % 5][col1] + matrix[(row2 - 1) % 5][col2]
        else:  # Different row and column
            decrypted_text += matrix[row1][col2] + matrix[row2][col1]

    return decrypted_text
#Hill CIpher encryption
import numpy as np

def prepare_input(text, n):
    text = text.replace(" ", "").upper()
    if len(text) % n != 0:
        text += 'X' * (n - (len(text) % n))
    return text

def text_to_matrix(text, n):
    matrix = []
    for char in text:
        matrix.append(ord(char) - 65)
    return np.array(matrix).reshape(-1, n)

def matrix_to_text(matrix):
    text = ""
    for row in matrix:
        for val in row:
            text += chr(val + 65)
    return text

def hill_cipher_encrypt(plaintext, key):
    n = int(np.sqrt(len(key)))
    plaintext = prepare_input(plaintext, n)
    key_matrix = np.array(text_to_matrix(key, n))
    plaintext_matrix = text_to_matrix(plaintext, n)

    encrypted_matrix = np.dot(plaintext_matrix, key_matrix) % 26
    encrypted_text = matrix_to_text(encrypted_matrix)
    return encrypted_text

#hill cipher decryption
import numpy as np

def prepare_input(text, n):
    text = text.replace(" ", "").upper()
    if len(text) % n != 0:
        text += 'X' * (n - (len(text) % n))
    return text

def text_to_matrix(text, n):
    matrix = []
    for char in text:
        matrix.append(ord(char) - 65)
    return np.array(matrix).reshape(-1, n)

def matrix_to_text(matrix):
    text = ""
    for row in matrix:
        for val in row:
            text += chr(val + 65)
    return text

def find_mod_inverse(det, m):
    for i in range(1, m):
        if (det * i) % m == 1:
            return i
    return None

def hill_cipher_decrypt(ciphertext, key):
    n = int(np.sqrt(len(key)))
    ciphertext = prepare_input(ciphertext, n)
    key_matrix = np.array(text_to_matrix(key, n))
    ciphertext_matrix = text_to_matrix(ciphertext, n)

    det = np.linalg.det(key_matrix)
    m = 26

    det_inverse = find_mod_inverse(int(det), m)
    if det_inverse is None:
        return "Error: Determinant has no inverse in mod 26"

    key_inverse = np.round(det_inverse * np.linalg.det(key_matrix) * np.linalg.inv(key_matrix)).astype(int) % 26

    decrypted_matrix = np.dot(ciphertext_matrix, key_inverse) % 26
    decrypted_text = matrix_to_text(decrypted_matrix)
    return decrypted_text

#rail fence enryption
def rail_fence_encrypt(text, rails):
    fence = [[] for _ in range(rails)]
    direction = 1  # Direction indicator for zigzag pattern
    row = 0

    for char in text:
        fence[row].append(char)
        row += direction

        # Change direction when reaching the top or bottom rail
        if row == 0 or row == rails - 1:
            direction *= -1

    encrypted_text = ''.join([''.join(rail) for rail in fence])
    return encrypted_text
#rail fence decryption
def rail_fence_decrypt(text, rails):
    fence = [['\n' for _ in range(len(text))] for _ in range(rails)]
    direction = 1
    row, col = 0, 0

    for i in range(len(text)):
        if row == 0:
            direction = 1
        if row == rails - 1:
            direction = -1
        fence[row][col] = '*'
        col += 1
        row += direction

    index = 0
    for i in range(rails):
        for j in range(len(text)):
            if fence[i][j] == '*' and index < len(text):
                fence[i][j] = text[index]
                index += 1

    decrypted_text = ''
    row, col = 0, 0
    for i in range(len(text)):
        if row == 0:
            direction = 1
        if row == rails - 1:
            direction = -1
        if fence[row][col] != '*':
            decrypted_text += fence[row][col]
            col += 1
        row += direction

    return decrypted_text
#coulumnar encryption
def columnar_encrypt(plaintext, keyword):
    plaintext = ''.join(filter(str.isalpha, plaintext.upper()))
    keyword_order = ''.join(sorted(set(keyword), key=keyword.index))
    rows = -(-len(plaintext) // len(keyword))
    matrix = ['' for _ in range(rows)]

    for i, char in enumerate(plaintext):
        matrix[i // len(keyword)] += char

    ordered_columns = [matrix[keyword_order.index(char)] for char in sorted(keyword_order)]
    ciphertext = ''.join(ordered_columns)
    return ciphertext
#columnar decryption
def columnar_decrypt(ciphertext, keyword):
    keyword_order = ''.join(sorted(set(keyword), key=keyword.index))
    rows = -(-len(ciphertext) // len(keyword))
    num_cols = len(keyword)
    num_rows = -(-len(ciphertext) // num_cols)
    empty_cells = (num_cols * num_rows) - len(ciphertext)
    plaintext = [''] * num_rows

    col_widths = [num_rows - (i < empty_cells) for i in range(num_cols)]
    col = 0
    for col_width in col_widths:
        for row in range(col_width):
            plaintext[row] += ciphertext[col]
            col += 1

    ordered_keyword = sorted(keyword)
    keyword_indices = [keyword.index(ordered_keyword[i]) for i in range(num_cols)]
    reordered_plaintext = [''] * num_rows

    for i, index in enumerate(keyword_indices):
        reordered_plaintext[index] = plaintext[i]

    decrypted_text = ''.join(reordered_plaintext)
    return decrypted_text



# Function to perform encryption or decryption based on selected algorithm
# ... (Previous code)

# Function to perform encryption or decryption based on selected algorithm
def perform_operation():
    selected_operation = combo_operation.get()
    selected_algorithm = combo_algorithm.get()
    
    plaintext = entry_text.get()
    key = entry_key.get()
    
    if selected_operation == "Encryption":
        if selected_algorithm == "Caesar Cipher":
            shift_key = int(key)
            encrypted_text = caesar_cipher_encrypt(plaintext, shift_key)
            label_result.config(text=f"Encrypted text: {encrypted_text}")
            return
        elif selected_algorithm == "Mono Alphabetic Cipher":
            encrypted_text = monoalphabetic_encrypt(plaintext, key)
            label_result.config(text=f"Encrypted text: {encrypted_text}")
            return
        elif selected_algorithm == "Poly Alphabetic Cipher":
            encrypted_text = vigenere_cipher_encrypt(plaintext, key)
            label_result.config(text=f"Encrypted text: {encrypted_text}")
            return
        elif selected_algorithm == "Play fair Cipher":
            encrypted_text = encrypt_playfair(plaintext, key)
            label_result.config(text=f"Encrypted text: {encrypted_text}")
            return
        elif selected_algorithm == "Hill Cipher":
            encrypted_text = hill_cipher_encrypt(plaintext, key)
            label_result.config(text=f"Encrypted text: {encrypted_text}")
            return
        elif selected_algorithm == "Rail Fence Cipher":
            encrypted_text = rail_fence_encrypt(plaintext, int(key))
            label_result.config(text=f"Encrypted text: {encrypted_text}")
            return
        elif selected_algorithm == "Columnar Cipher":
            encrypted_text = columnar_encrypt(plaintext, key)
            label_result.config(text=f"Encrypted text: {encrypted_text}")
            return

    elif selected_operation == "Decryption":
        if selected_algorithm == "Caesar Cipher":
            shift_key = int(key)
            decrypted_text = caesar_cipher_decrypt(plaintext, shift_key)
            label_result.config(text=f"Decrypted text: {decrypted_text}")
            return
        elif selected_algorithm == "Mono Alphabetic Cipher":
            decrypted_text = monoalphabetic_decrypt(plaintext, key)
            label_result.config(text=f"Decrypted text: {decrypted_text}")
            return
        elif selected_algorithm == "Poly Alphabetic Cipher":
            decrypted_text = vigenere_cipher_decrypt(plaintext, key)
            label_result.config(text=f"Decrypted text: {decrypted_text}")
            return
        elif selected_algorithm == "Play fair Cipher":
            decrypted_text = decrypt_playfair(plaintext, key)
            label_result.config(text=f"Decrypted text: {decrypted_text}")
            return
        elif selected_algorithm == "Hill Cipher":
            decrypted_text = hill_cipher_decrypt(plaintext, key)
            label_result.config(text=f"Decrypted text: {decrypted_text}")
            return
        elif selected_algorithm == "Rail Fence Cipher":
            decrypted_text = rail_fence_decrypt(plaintext, int(key))
            label_result.config(text=f"Decrypted text: {decrypted_text}")
            return
        elif selected_algorithm == "Columnar Cipher":
            decrypted_text = columnar_decrypt(plaintext, key)
            label_result.config(text=f"Decrypted text: {decrypted_text}")
            return

# ... (Rest of the code remains unchanged)


# Create main window
root = tk.Tk()
root.title("Encryption and Decryption")

# Create a frame for better organization
main_frame = ttk.Frame(root, padding=20)
main_frame.grid(row=0, column=0)

# Create label and entry for input text
label_text = ttk.Label(main_frame, text="Enter Text:")
label_text.grid(row=0, column=0, padx=5, pady=5, sticky="w")
entry_text = ttk.Entry(main_frame, width=40)
entry_text.grid(row=0, column=1, padx=5, pady=5)

# Create label and entry for input key
label_key = ttk.Label(main_frame, text="Enter Key:")
label_key.grid(row=1, column=0, padx=5, pady=5, sticky="w")
entry_key = ttk.Entry(main_frame, width=40)
entry_key.grid(row=1, column=1, padx=5, pady=5)

# Create dropdown for selecting encryption/decryption
label_operation = ttk.Label(main_frame, text="Select Operation:")
label_operation.grid(row=2, column=0, padx=5, pady=5, sticky="w")
operations = ["Encryption", "Decryption"]
combo_operation = ttk.Combobox(main_frame, values=operations, state="readonly", width=15)
combo_operation.grid(row=2, column=1, padx=5, pady=5)
combo_operation.current(0)

# Create dropdown for selecting algorithms
label_algorithm = ttk.Label(main_frame, text="Select Algorithm:")
label_algorithm.grid(row=3, column=0, padx=5, pady=5, sticky="w")
algorithms = ["Caesar Cipher", "Mono Alphabetic Cipher","Poly Alphabetic Cipher","Play fair Cipher","Hill Cipher","Rail Fence Cipher","Columnar Cipher"]
combo_algorithm = ttk.Combobox(main_frame, values=algorithms, state="readonly", width=25)
combo_algorithm.grid(row=3, column=1, padx=5, pady=5)
combo_algorithm.current(0)

# Create button to perform operation
button_execute = ttk.Button(main_frame, text="Execute", command=perform_operation)
button_execute.grid(row=4, column=0, columnspan=2, pady=10)

# Create label to display result
label_result = ttk.Label(main_frame, text="")
label_result.grid(row=5, column=0, columnspan=2, pady=5)

root.mainloop()
