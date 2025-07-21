import os
import struct
import re

# Define the custom base64 alphabet
alphabet = r"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!#$%&()*+,-./:;<=>?@[]_^`{~}"


def int_to_base64(n, alphabet):
    """Encodes an integer into a fixed 2-character base64 string using a custom alphabet."""
    original = n
    warning = None

    if n < 0:
        n = 2048 - n  # Map negative numbers to [2048, 4095] range

    if n < 0:
        n = 0
        warning = f"Value {original} below minimum -2048, clamped to 0"
    elif n > 4095:
        n = 4095
        warning = f"Value {original} above maximum 4095, clamped to 4095"

    first_6_bits = (n >> 6) & 0x3F
    second_6_bits = n & 0x3F
    return alphabet[first_6_bits] + alphabet[second_6_bits], warning


def int_to_base64_24bit(n, alphabet):
    """Encodes a 24-bit integer into 4 base64 characters."""
    # used for the final bias in the NNUE, as it's usually larger than 2048
    # (it's multiplied by QA * QB during quantization)

    original = n
    warning = None

    # Handle negative values using two's complement
    if n < 0:
        n = (1 << 24) + n  # Map to 24-bit unsigned equivalent

    # Clamp to valid 24-bit range
    if n < 0:
        n = 0
        warning = f"Value {original} below minimum -8,388,608, clamped to 0"
    elif n > (1 << 24) - 1:
        n = (1 << 24) - 1
        warning = f"Value {original} above maximum 8,388,607, clamped to 8,388,607"

    # Extract 6-bit segments
    b1 = (n >> 18) & 0x3F
    b2 = (n >> 12) & 0x3F
    b3 = (n >> 6) & 0x3F
    b4 = n & 0x3F

    return (
            alphabet[b1] +
            alphabet[b2] +
            alphabet[b3] +
            alphabet[b4]
    ), warning


def read_raw_bin(path, is_quantized):
    """Reads a binary file containing quantized (int16) or unquantized (float32) weights."""
    with open(path, 'rb') as file:
        data = file.read()

    if is_quantized:
        return [struct.unpack('<h', data[i:i + 2])[0] for i in range(0, len(data), 2)]
    else:
        return [struct.unpack('<f', data[i:i + 4])[0] for i in range(0, len(data), 4)]


def save_to_file(filename, weights, directory="output"):
    """Writes a list of weights to a file inside the specified directory."""
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, filename)

    with open(file_path, "w") as file:
        file.write("\n".join(map(str, weights)))


def create_base64_metadata(H, b, O, c, alphabet, network_name, directory="output"):
    """Creates the combined metadata and base64 encoded weights file with enhanced format."""
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, "network_base64.txt")

    # Create enhanced metadata string
    input_size = len(H) // len(b)
    metadata = (
        f"name={network_name},"
        f"input={input_size},"
        f"hidden={len(b)},"
        f"output={len(c)},"
        f"version=2,"  # Format version identifier
        f"bias_encoding=24bit"  # Special encoding for last layer
    )

    # Convert all components to base64 with warnings
    warnings = []

    print("\nEncoding components to base64...")
    H_base64, h_warnings = "", []
    for n in H:
        encoded, warning = int_to_base64(n, alphabet)
        H_base64 += encoded
        if warning: h_warnings.append(warning)

    b_base64, b_warnings = "", []
    for n in b:
        encoded, warning = int_to_base64(n, alphabet)
        b_base64 += encoded
        if warning: b_warnings.append(warning)

    O_base64, o_warnings = "", []
    for n in O:
        encoded, warning = int_to_base64(n, alphabet)
        O_base64 += encoded
        if warning: o_warnings.append(warning)

    # Special 24-bit encoding for last bias layer
    c_base64, c_warnings = "", []
    for n in c:
        encoded, warning = int_to_base64_24bit(n, alphabet)
        c_base64 += encoded
        if warning: c_warnings.append(warning)

    # Report warnings
    if h_warnings:
        print(f"  WARNING: {len(h_warnings)} clamping issues in H weights")
        warnings.extend(h_warnings[:3])  # Show first 3
        if len(h_warnings) > 3: warnings.append(f"... and {len(h_warnings) - 3} more")

    if b_warnings:
        print(f"  WARNING: {len(b_warnings)} clamping issues in b biases")
        warnings.extend(b_warnings[:3])
        if len(b_warnings) > 3: warnings.append(f"... and {len(b_warnings) - 3} more")

    if o_warnings:
        print(f"  WARNING: {len(o_warnings)} clamping issues in O weights")
        warnings.extend(o_warnings[:3])
        if len(o_warnings) > 3: warnings.append(f"... and {len(o_warnings) - 3} more")

    if c_warnings:
        print(f"  WARNING: {len(c_warnings)} clamping issues in c biases")
        warnings.extend(c_warnings[:3])
        if len(c_warnings) > 3: warnings.append(f"... and {len(c_warnings) - 3} more")

    # Combine all components with enhanced metadata format
    combined = f"[{metadata}]|{H_base64}|{b_base64}|{O_base64}|{c_base64}"

    with open(file_path, "w") as file:
        file.write(combined)

    print(f"Created combined base64 file: {file_path}")

    # Save warnings to file
    if warnings:
        warn_path = os.path.join(directory, "base64_warnings.txt")
        with open(warn_path, "w") as f:
            f.write("\n".join(warnings))
        print(f"  Saved {len(warnings)} warnings to {warn_path}")


def write_data_to_bc_format(weights, is_quantized, bin_path):
    """Splits weights into sections and saves them with base64 conversion."""
    input_size = 768
    hidden_size = 256
    output_size = 1

    h_size = input_size * hidden_size
    b_size = hidden_size
    o_size = 2 * hidden_size
    c_size = output_size
    expected_total = h_size + b_size + o_size + c_size

    if len(weights) != expected_total:
        print(f"\nWARNING: Weight count mismatch!")
        print(f"  Expected: {expected_total} weights")
        print(f"  Found:    {len(weights)} weights")

        if len(weights) > expected_total:
            print(f"  Trimming {len(weights) - expected_total} extra weights")
            weights = weights[:expected_total]
        else:
            print(f"  Padding with {expected_total - len(weights)} zeros")
            weights += [0] * (expected_total - len(weights))

    # Split into components
    idx = 0
    H = weights[idx: idx + h_size]
    idx += h_size
    b = weights[idx: idx + b_size]
    idx += b_size
    O = weights[idx: idx + o_size]
    idx += o_size
    c = weights[idx: idx + c_size]
    idx += c_size

    print("\nParameter counts:")
    print(f"  NN ACC in Weights: {len(H)}")
    print(f"  NN Bias 1 (Accumulator): {len(b)}")
    print(f"  NN Weights 1: {len(O)}")
    print(f"  NN Bias 2: {len(c)}")

    # Save plaintext versions with original filenames
    save_to_file("NN ACC in Weights.txt", H)
    save_to_file("NN Bias 1 (Accumulator).txt", b)
    save_to_file("NN Weights 1.txt", O)
    save_to_file("NN Bias 2.txt", c)

    # Create combined base64 file
    if is_quantized:
        # Extract network name from bin_path
        network_name = os.path.splitext(os.path.basename(bin_path))[0]
        create_base64_metadata(H, b, O, c, alphabet, network_name)
    else:
        print("\nWARNING: Skipping base64 conversion for unquantized weights")
        print("  Base64 format is only supported for quantized networks")


def verify_network_structure(weights, is_quantized):
    """Performs basic validation of network structure"""
    if not is_quantized:
        print("  Skipping quantized-specific validation")
        return True

    # Check for all-zero weights
    zero_count = sum(1 for w in weights if w == 0)
    zero_percent = (zero_count / len(weights)) * 100
    if zero_percent > 10:
        print(f"\nWARNING: High percentage of zero weights ({zero_percent:.2f}%)")

    # Check value ranges
    min_val = min(weights)
    max_val = max(weights)
    print(f"  Value range: [{min_val}, {max_val}]")

    if min_val < -2048 or max_val > 2047:
        print("  WARNING: Values outside [-2048, 2047] range detected")
        print("    These will be clamped during base64 encoding")

    return True


def decode_base64_12bit(s, alphabet):
    """Decodes a 2-character base64 string to a 12-bit integer using your custom scheme."""
    if len(s) != 2:
        raise ValueError("12-bit encoding requires 2 characters")

    first_char = s[0]
    second_char = s[1]
    first_6bits = alphabet.index(first_char)
    second_6bits = alphabet.index(second_char)
    total = (first_6bits << 6) | second_6bits

    # Convert using your custom scheme: if value >= 2048, it's negative
    if total >= 2048:
        return 2048 - total
    return total

def decode_base64_24bit(s, alphabet):
    """Decodes a 4-character base64 string to a 24-bit integer using consistent scheme."""
    if len(s) != 4:
        raise ValueError("24-bit encoding requires 4 characters")

    total = 0
    for char in s:
        total = (total << 6) | alphabet.index(char)

    # Apply the same negative conversion scheme as 12-bit
    midpoint = 1 << 23  # 8,388,608
    if total >= midpoint:
        return midpoint - total
    return total

def parse_metadata(meta_str):
    """Parses metadata string into a dictionary."""
    metadata = {}
    # Remove brackets if present
    clean_str = meta_str.strip("[]")

    # Handle potential comma in network name
    parts = re.split(r",(?=\w+=)", clean_str)

    for part in parts:
        if "=" in part:
            key, value = part.split("=", 1)
            metadata[key] = value
    return metadata


def verify_base64_file(file_path, original_H, original_b, original_O, original_c, alphabet):
    """Verifies the integrity of the base64 encoded file."""
    print("\nVerifying base64 file integrity...")
    try:
        with open(file_path, 'r') as f:
            data = f.read().strip()
    except FileNotFoundError:
        print(f"  ERROR: File not found at {file_path}")
        return False

    # Find metadata section
    if not data.startswith('['):
        print("  ERROR: Metadata missing opening bracket")
        return False

    end_meta = data.find(']')
    if end_meta == -1:
        print("  ERROR: Metadata missing closing bracket")
        return False

    # Extract metadata and the rest of the data
    meta_str = data[1:end_meta]
    weight_data = data[end_meta + 1:]

    # Validate pipe separator after metadata
    if not weight_data.startswith('|'):
        print("  ERROR: Missing pipe separator after metadata")
        return False

    # Split the remaining components
    parts = weight_data[1:].split('|')
    if len(parts) != 4:
        print(f"  ERROR: Expected 4 weight components, got {len(parts)}")
        print(f"  Data structure: metadata]{weight_data}")
        print(f"  Split parts: {parts}")
        return False

    H_str, b_str, O_str, c_str = parts

    # Parse metadata
    try:
        metadata = parse_metadata(meta_str)
        print(f"  Metadata: {metadata}")
    except Exception as e:
        print(f"  ERROR parsing metadata: {e}")
        return False

    # Verify component lengths
    expected_H_len = len(original_H) * 2
    expected_b_len = len(original_b) * 2
    expected_O_len = len(original_O) * 2
    expected_c_len = len(original_c) * 4

    valid = True

    # Check H weights
    if len(H_str) != expected_H_len:
        print(f"  ERROR: H length mismatch: expected {expected_H_len}, got {len(H_str)}")
        valid = False

    # Check b biases
    if len(b_str) != expected_b_len:
        print(f"  ERROR: b length mismatch: expected {expected_b_len}, got {len(b_str)}")
        valid = False

    # Check O weights
    if len(O_str) != expected_O_len:
        print(f"  ERROR: O length mismatch: expected {expected_O_len}, got {len(O_str)}")
        valid = False

    # Check c biases
    if len(c_str) != expected_c_len:
        print(f"  ERROR: c length mismatch: expected {expected_c_len}, got {len(c_str)}")
        valid = False

    if not valid:
        return False

    print("  Decoding and comparing values...")

    def verify_component(name, original_values, base64_str, bit_size):
        """Helper function to verify a single component"""
        mismatches = 0
        char_count = 2 if bit_size == 12 else 4
        decode_func = decode_base64_12bit if bit_size == 12 else decode_base64_24bit
        midpoint = 2048 if bit_size == 12 else (1 << 23)  # 8,388,608 for 24-bit

        for i in range(len(original_values)):
            # Extract encoded substring
            start = i * char_count
            encoded = base64_str[start:start + char_count]

            # Decode value
            try:
                decoded = decode_func(encoded, alphabet)
            except Exception as e:
                print(f"    ERROR decoding {name} at index {i}: {e}")
                decoded = 0
                mismatches += 1
                continue

            # Calculate expected value with clamping
            original_val = original_values[i]
            if original_val < -midpoint:
                expected = -midpoint
            elif original_val > midpoint - 1:
                expected = midpoint - 1
            else:
                expected = original_val

            # Compare
            if decoded != expected:
                if mismatches < 5:  # Show first 5 mismatches
                    print(f"    {name} mismatch at {i}:")
                    print(f"      Decoded: {decoded}")
                    print(f"      Expected: {expected} (after clamping)")
                    print(f"      Original: {original_val}")
                    print(f"      Encoded: {encoded}")
                    # Show alphabet indices for debugging
                    indices = [alphabet.index(c) for c in encoded]
                    print(f"      Alphabet indices: {indices}")
                mismatches += 1

        if mismatches:
            print(f"  {name}: {mismatches}/{len(original_values)} mismatches")
        return mismatches == 0

    # Verify each component
    valid = True
    print("  Verifying H weights (12-bit)...")
    valid &= verify_component("H", original_H, H_str, 12)

    print("  Verifying b biases (12-bit)...")
    valid &= verify_component("b", original_b, b_str, 12)

    print("  Verifying O weights (12-bit)...")
    valid &= verify_component("O", original_O, O_str, 12)

    print("  Verifying c biases (24-bit)...")
    valid &= verify_component("c", original_c, c_str, 24)

    if valid:
        print("  All components verified successfully!")
    else:
        print("  Verification failed with mismatches")

    return valid

def main():
    os.makedirs("output", exist_ok=True)

    # Determine which file to process
    bin_path = "bin/network.bin" if os.path.exists("bin/network.bin") else "bin/quantised.bin"
    is_quantized = "quantised.bin" in bin_path

    print(f"Processing {'quantized' if is_quantized else 'raw'} network: {bin_path}")
    weights = read_raw_bin(bin_path, is_quantized)

    # Basic validation
    print("\nVerifying network structure...")
    if not verify_network_structure(weights, is_quantized):
        print("ERROR: Network validation failed. Aborting processing")
        return

    # Save full weights as plaintext
    save_to_file("all_values_translated_res.txt", weights)

    # Process weights into components
    write_data_to_bc_format(weights, is_quantized, bin_path)

    # Verify base64 file if quantized
    if is_quantized:
        print("\nVerifying exported base64 file...")
        file_path = os.path.join("output", "network_base64.txt")

        # Reconstruct original components
        input_size = 768
        hidden_size = 256
        output_size = 1
        h_size = input_size * hidden_size
        b_size = hidden_size
        o_size = 2 * hidden_size
        c_size = output_size

        idx = 0
        original_H = weights[idx: idx + h_size]
        idx += h_size
        original_b = weights[idx: idx + b_size]
        idx += b_size
        original_O = weights[idx: idx + o_size]
        idx += o_size
        original_c = weights[idx: idx + c_size]

        success = verify_base64_file(
            file_path,
            original_H,
            original_b,
            original_O,
            original_c,
            alphabet
        )

        if success:
            print("Base64 file verification: SUCCESS")
        else:
            print("Base64 file verification: FAILED")

    print("\nConversion complete. All files saved to 'output' directory")
    print("Summary of files created:")
    print("  - NN ACC in Weights.txt")
    print("  - NN Bias 1 (Accumulator).txt")
    print("  - NN Weights 1.txt")
    print("  - NN Bias 2.txt")
    print("  - all_values_translated_res.txt")
    if is_quantized:
        print("  - network_base64.txt (combined metadata + base64)")
    print("\nCheck 'base64_warnings.txt' if any clamping issues occurred")


if __name__ == "__main__":
    main()