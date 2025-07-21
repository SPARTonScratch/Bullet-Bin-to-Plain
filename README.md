# Bullet-Bin-to-Plain
Tools for converting a simple single perspective network from the [Bullet NNUE trainer](https://github.com/jw1912/bullet) to plaintext and the "portable NNUE" format.

While most engines would directly use the .bin file produced by [Bullet](https://github.com/jw1912/bullet), it sometimes may be useful to get plaintext for debugging, and the "portable NNUE" format is good for Scratch Chess Engines, such as [White Dove](https://github.com/SPARTonScratch/White-Dove-Chess-Engine).

## File Structure
`[name=NetworkName,input=768,hidden=256,output=1,version=2,bias_encoding=24bit]|H...|b...|O...|c...`

## Metadata (Square Brackets)
Comma-separated key-value pairs:
- `name`: Network identifier (spaces/special chars allowed)
- `input`: Input layer size (e.g., 768 features)
- `hidden`: Hidden layer size (e.g., 256 neurons)
- `output`: Output layer size (e.g., 1 value)
- `version`: Format version (current: 2)
- `bias_encoding`: Precision for last bias (`24bit`)

## Weight Components (Pipe-Separated)
| Component | Description                     | Size                  | Encoding |
|-----------|---------------------------------|-----------------------|----------|
| `H`       | Input weights                   | input × hidden        | 12-bit   |
| `b`       | Hidden layer biases             | hidden                | 12-bit   |
| `O`       | Output weights                  | 2 × hidden            | 12-bit   |
| `c`       | Output bias (high precision)    | output                | 24-bit   |

## Custom Base64 Encoding
**Alphabet:**  
```ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!#$%&()*+,-./:;<=>?@[]_^`{~}```

**Value Ranges:**
- **12-bit values** (H, b, O):
  - Positive: `0` to `2047`
  - Negative: `-1` to `-2048`
  - Encoded as **2 characters**
  
- **24-bit values** (c):
  - Positive: `0` to `8,388,607`
  - Negative: `-1` to `-8,388,608`
  - Encoded as **4 characters**

**Negative Handling:**  
Negative values `n` are transformed:  
`encoded_value = midpoint - n`  
Where midpoint is `2048` (12-bit) or `8,388,608` (24-bit)

## Example Encoding
| Value | Type   | Encoded | Calculation             |
|-------|--------|---------|-------------------------|
| 5     | 12-bit | `AF`    | `A`=0, `F`=5 → (0×64)+5 |
| -3    | 12-bit | `DK`    | 2048 - (-3) = 2051 → `D`=3, `K`=11 |
| 3725  | 24-bit | `AA_N`  | `A`=0, `A`=0, `_`=58, `N`=13 → (58×64)+13=3725 |
| -1000 | 24-bit | `~~zF`  | 8,388,608 - (-1000) = 8,389,608 → converted to 4 chars |

## Important Notes
1. **Byte Order**: Little-endian (`<h`/`<f`) for binary files
2. **Clamping**: Values outside ranges are clamped to min/max
3. **Compatiblity**: 
   - `version=1`: Original 12-bit only format
   - `version=2`: Adds 24-bit final bias support
4. **Character Set**: 64-character alphabet covers all possible 6-bit values

## Credits

Thanks to @jw for creating the [Bullet NNUE trainer](https://github.com/jw1912/bullet).
