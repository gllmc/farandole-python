from typing import Optional
from dataclasses import dataclass
from hashlib import shake_256
import time
import secrets

import numpy as np

from constants import n, k, q, kappa, field, beta, gamma, eta
from noise import generate_noise
from ntt import PolyRingRqNtt


@dataclass
class SecretKey:
    S_hat: np.ndarray
    B_inv_hat: np.ndarray
    A_hat: np.ndarray
    z: bytes


@dataclass
class PublicKey:
    A_hat: np.ndarray
    B_hat: np.ndarray


@dataclass
class Ciphertext:
    c1: np.ndarray
    c2: np.ndarray


type SymmetricKey = bytes


ntt = PolyRingRqNtt()


def bit_pack_vector(vector: np.ndarray) -> bytes:
    coefficient_bit_length = (q - 1).bit_length()
    out = bytearray()
    buffer = 0
    bits_in_buffer = 0
    for coefficient in vector.flatten():
        buffer |= (int(coefficient) % q) << bits_in_buffer
        bits_in_buffer += coefficient_bit_length
        while bits_in_buffer >= 8:
            out.append(buffer & 0xFF)
            buffer >>= 8
            bits_in_buffer -= 8
    if bits_in_buffer > 0:
        out.append(buffer)
    return bytes(out)


def compute_shared_key(e1: np.ndarray, e2: np.ndarray, r: np.ndarray) -> bytes:
    hasher = shake_256()
    hasher.update(bit_pack_vector(e1))
    hasher.update(bit_pack_vector(e2))
    hasher.update(bit_pack_vector(r))
    return hasher.digest(kappa)


def compute_implicit_rejection(z: bytes, c1: np.ndarray, c2: np.ndarray) -> bytes:
    hasher = shake_256()
    hasher.update(z)
    hasher.update(bit_pack_vector(c1))
    hasher.update(bit_pack_vector(c2))
    return hasher.digest(kappa)


def norm_squared(x: np.ndarray) -> int:
    x_int = np.array(x, dtype=np.int64)
    x_centered = np.where(x_int > q // 2, x_int - q, x_int)
    return int(np.sum(x_centered**2))


def keygen() -> tuple[SecretKey, PublicKey]:
    B_inv_hat = None
    while B_inv_hat is None:
        A = field(np.random.randint(0, q, size=(n, k, k), dtype=np.int64))
        S = generate_noise((n, k, k))
        E = generate_noise((n, k, k))
        A_hat, S_hat, E_hat = ntt.ntt(A), ntt.ntt(S), ntt.ntt(E)
        B_hat = (2 * eta + 1) * (S_hat @ A_hat + E_hat)
        try:
            B_inv_hat = field.Zeros((n, k, k))
            for i in range(n):
                B_inv_hat[i] = np.linalg.inv(B_hat[i])
        except np.linalg.LinAlgError:
            B_inv_hat = None
    z = secrets.token_bytes(gamma)
    return SecretKey(S_hat=S_hat, B_inv_hat=B_inv_hat, A_hat=A_hat, z=z), PublicKey(
        A_hat=A_hat, B_hat=B_hat
    )


def encaps(pk: PublicKey) -> tuple[SymmetricKey, Ciphertext]:
    r = generate_noise((n, k, 1))
    e1 = generate_noise((n, k, 1))
    e2 = generate_noise((n, k, 1))

    r_hat, e1_hat, e2_hat = ntt.ntt(r), ntt.ntt(e1), ntt.ntt(e2)

    c1 = pk.A_hat @ r_hat + e1_hat
    c2 = pk.B_hat @ r_hat + e2_hat

    key = compute_shared_key(e1, e2, r)

    return key, Ciphertext(c1=c1, c2=c2)


def decaps(
    sk: SecretKey,
    ct: Ciphertext,
) -> Optional[SymmetricKey]:
    p = 2 * eta + 1
    mu = ntt.intt(ct.c2 - p * sk.S_hat @ ct.c1)
    mu_int = np.array(mu, dtype=np.int64)
    mu_centered = np.where(mu_int > q // 2, mu_int - q, mu_int)
    val = mu_centered % p
    val_centered = np.where(val > p // 2, val - p, val)
    e2 = field(val_centered % q)
    e2_hat = ntt.ntt(e2)

    r_hat = sk.B_inv_hat @ (ct.c2 - e2_hat)
    e1_hat = ct.c1 - sk.A_hat @ r_hat
    r, e1 = ntt.intt(r_hat), ntt.intt(e1_hat)

    flag = max(norm_squared(e1), norm_squared(r)) > beta

    key = (
        compute_shared_key(e1, e2, r)
        if not flag
        else compute_implicit_rejection(sk.z, ct.c1, ct.c2)
    )

    return key


if __name__ == "__main__":
    NB_ITERATIONS = 10

    keygen_total_time = 0
    encaps_total_time = 0
    decaps_total_time = 0
    total_total_time = 0

    for i in range(NB_ITERATIONS):
        start_time = time.perf_counter()

        sk, pk = keygen()
        keygen_time = time.perf_counter()

        key, ct = encaps(pk)
        encaps_time = time.perf_counter()

        decapsulated_key = decaps(sk, ct)
        decaps_time = time.perf_counter()

        assert (
            decapsulated_key == key
        ), "Error: decapsulated key does not match original key."

        total_time = decaps_time - start_time
        t_keygen = keygen_time - start_time
        t_encaps = encaps_time - keygen_time
        t_decaps = decaps_time - encaps_time

        keygen_total_time += t_keygen
        encaps_total_time += t_encaps
        decaps_total_time += t_decaps
        total_total_time += total_time

    print(f"\nAll {NB_ITERATIONS} iterations were successful. Average times:")
    print("-" * 40)
    print(f"Key generation  : {keygen_total_time / NB_ITERATIONS * 1000:>8.2f} ms")
    print(f"Encapsulation   : {encaps_total_time / NB_ITERATIONS * 1000:>8.2f} ms")
    print(f"Decapsulation   : {decaps_total_time / NB_ITERATIONS * 1000:>8.2f} ms")
    print("-" * 40)
    print(f"Total time      : {total_total_time / NB_ITERATIONS * 1000:>8.2f} ms")
