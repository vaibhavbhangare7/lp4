import heapq


def count_frequency(data):
    frequency = {}
    for char in data:
        if char in frequency:
            frequency[char] += 1
        else:
            frequency[char] = 1
    return frequency

class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(frequency):
    heap = [Node(char, freq) for char, freq in frequency.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)
    
    return heap[0]

def generate_codes(node, current_code="", code_map={}):
    if node is None:
        return

    if node.char is not None:
        code_map[node.char] = current_code

    generate_codes(node.left, current_code + "0", code_map)
    generate_codes(node.right, current_code + "1", code_map)

    return code_map

def huffman_encode(data, code_map):
    return ''.join(code_map[char] for char in data)

def huffman_decode(encoded_data, root):
    decoded_output = []
    current_node = root
    
    for bit in encoded_data:
        current_node = current_node.left if bit == '0' else current_node.right
        
        if current_node.char is not None:
            decoded_output.append(current_node.char)
            current_node = root

    return ''.join(decoded_output)



data = "Vaibhav"
frequency = count_frequency(data)

root = build_huffman_tree(frequency)

code_map = generate_codes(root)

encoded_data = huffman_encode(data, code_map)
print("Encoded Data:", encoded_data)

decoded_data = huffman_decode(encoded_data, root)
print("Decoded Data:", decoded_data)


