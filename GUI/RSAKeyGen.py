

import random
from math import gcd

# quick function to check if a number is prime
def is_prime(n):
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

# makes a random prime number with the given bit size
def generate_prime(bits=8):
    while True:
        num = random.getrandbits(bits)
        if is_prime(num):
            return num

# finds the modular inverse of e mod phi
def mod_inverse(e, phi):
    def egcd(a, b):
        if a == 0:
            return (b, 0, 1)
        else:
            g, y, x = egcd(b % a, a)
            return (g, x - (b // a) * y, y)
    
    g, x, _ = egcd(e, phi)
    if g != 1:
        raise Exception("No modular inverse found.")
    return x % phi

# generates public and private RSA keys
def generate_rsa_keys(bits=8):
    print(f"Generating {bits}-bit RSA keypair...")

    # step 1: pick two random primes
    p = generate_prime(bits)
    q = generate_prime(bits)
    while p == q:
        q = generate_prime(bits)

    n = p * q
    phi = (p - 1) * (q - 1)

    # step 2: pick e that has no common factor with phi
    e = random.randrange(2, phi)
    while gcd(e, phi) != 1:
        e = random.randrange(2, phi)

    # step 3: calculate d (the private key)
    d = mod_inverse(e, phi)

    public_key = (e, n)
    private_key = (d, n)

    print(f"p = {p}, q = {q}")
    print(f"n = {n}, phi = {phi}")
    print(f"Public key: {public_key}")
    print(f"Private key: {private_key}")

    # saves keys to text files so we can use them later
    with open("public_key.txt", "w") as f:
        f.write(str(public_key))
    with open("private_key.txt", "w") as f:
        f.write(str(private_key))
    print("\nKeys saved as public_key.txt and private_key.txt")

    return (e, d, n)



# run the generator
if __name__ == "__main__":
    generate_rsa_keys(bits=8)

