# https://ericmjl.github.io/essays-on-data-science/machine-learning/markov-models/#bayesian-inference-on-markov-models

# from scipy.stats import multinomial
import numba
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim


def equilibrium_distribution(p_transition) -> np.ndarray:
    n_states = p_transition.shape[0]
    A = np.vstack([p_transition.T - np.eye(n_states), np.ones(n_states)])
    pinv = np.linalg.pinv(A)
    p_eq = pinv[:, -1]
    p_eq = np.maximum(p_eq, 0)
    return p_eq / p_eq.sum()


def markov_sequence(
    p_init: np.ndarray | None, p_transition: np.ndarray, sequence_length: int
) -> np.ndarray:
    if p_init is None:
        p_init = equilibrium_distribution(p_transition)
    # initial_state = list(multinomial.rvs(1, p_init)).index(1)
    initial_state = np.random.choice(len(p_init), p=p_init)
    states = [initial_state]
    for _ in range(sequence_length - 1):
        p_tr = p_transition[states[-1]]
        # new_state = list(multinomial.rvs(1, p_tr)).index(1)
        new_state = np.random.choice(len(p_tr), p=p_tr)
        states.append(new_state)
    return np.array(states)


@numba.njit
def count_transitions(states, n_states):
    counts = np.zeros((n_states, n_states), dtype=np.float64)
    for k in range(len(states) - 1):
        i = states[k]
        j = states[k + 1]
        counts[i, j] += 1
    return counts


def get_p_transition(states, n_states) -> np.ndarray:
    matrix = count_transitions(states, n_states)
    sums = matrix.sum(axis=1, keepdims=True)
    p_transition = np.divide(matrix, sums, out=np.zeros_like(matrix), where=sums != 0)
    return p_transition


def chars_to_states(chars: str):
    b = chars.encode("ascii", errors="ignore")
    arr = np.frombuffer(b, dtype=np.uint8)
    mask = (arr >= 32) & (arr <= 126)
    return arr[mask].astype(np.int32) - 32


def file_to_states(path):
    with open(path, "rb") as f:
        b = f.read()
    non_printable = bytes(range(0, 32)) + bytes([127, *range(128, 256)])
    filtered = b.translate(None, non_printable)
    return np.frombuffer(filtered, dtype=np.uint8).astype(np.int32) - 32


def get_translation_tables(text):
    text_ = "".join(chr_ for chr_ in text if chr_.isascii() and chr_.isprintable())
    unq_chr_set = set(text_)
    unq_chr_lst = sorted(list(unq_chr_set), key=lambda x: ord(x))
    translation = {c: i for i, c in enumerate(unq_chr_lst)}
    translation_table_lst = []
    for i in range(256):
        chr_ = chr(i)
        if not (chr_.isascii() and chr_.isprintable() and chr_ in unq_chr_set):
            translation_table_lst.append(b"\x00")
            continue
        chr_idx = unq_chr_lst.index(chr_)
        translation_table_lst.append(chr_idx.to_bytes())
    translation_table = b"".join(translation_table_lst)
    delete_table_lst = []
    for i in range(256):
        chr_ = chr(i)
        if not (chr_.isascii() and chr_.isprintable() and chr_ in unq_chr_set):
            delete_table_lst.append(i.to_bytes())
    delete_table = b"".join(delete_table_lst)
    return translation, translation_table, delete_table, len(unq_chr_lst)


def translation_table_nice(table: bytes):
    nice = {}
    for i in range(256):
        b = table[i]
        if i != b"\x00":
            pass
        nice[chr(b)] = i
    return nice


def file_to_optimized_states(path):
    with open(path, "rb") as f:
        text = f.read()
    translation, translation_table, delete_table, n_unq_chrs = get_translation_tables(
        text.decode()
    )
    translated = text.translate(translation_table, delete_table)
    return (
        np.frombuffer(translated, dtype=np.uint8).astype(np.int32),
        translation,
        n_unq_chrs,
    )


def states_to_chars(states: np.ndarray):
    return "".join(chr(state + 32) for state in states)


print("loading states...")
states, translation, N_CHARS = file_to_optimized_states("data/norm/ezpz.txt")

print("computing transition matrix...")
p_transition = get_p_transition(states, n_states=N_CHARS)

# isascii() and isprintable() ords range from 32 to 126, for 95 total
# N_CHARS = 95
N_DIMS = 12  # 82  # hand optimized lol
RANGE = 4.0
# K = np.sqrt(N_DIMS * (RANGE**2)) / 7
K = 4
SPACE_IDX = ord(" ") - 32  # 0

p_mask = torch.from_numpy(np.sum(p_transition, axis=1))

embeddings = torch.rand(N_CHARS, N_DIMS, dtype=torch.float32) * RANGE - RANGE / 2
embeddings[0] = 0.0
embeddings.requires_grad_(True)

expected_distances = torch.from_numpy(
    K / np.pow(np.e, p_transition + p_transition.T)
).to(torch.float32)
torch.diagonal(expected_distances).zero_()


optimizer = optim.AdamW([embeddings], lr=1e-2)

for i in range(24000):
    optimizer.zero_grad()

    distances = torch.cdist(embeddings, embeddings, p=2)

    space_loss = torch.pow(embeddings[0], 2).mean() / 1000
    mse_loss = (
        F.mse_loss(distances, expected_distances, reduction="none") * p_mask
    ).mean()
    loss = mse_loss + space_loss
    loss.backward()
    optimizer.step()

    if i % 1000 == 1000 - 1:
        print(f"epoch: {i + 1} loss: {loss.item()}")

embeddings.detach_()
torch.set_printoptions(profile="default", sci_mode=False, threshold=100)
print(embeddings)
print("std\t", embeddings.std().item())
print("min\t", embeddings.min().item())
print("max\t", embeddings.max().item())
print("mean\t", embeddings.mean().item())

embeddings.clamp_(-2.0, 2.0)
distances = torch.cdist(embeddings, embeddings, p=2)
space_loss = torch.pow(embeddings[0], 2).mean() / 1000
mse_loss = (F.mse_loss(distances, expected_distances, reduction="none") * p_mask).mean()
loss = mse_loss + space_loss

print("clipped loss:", loss.item())

save = {
    "embeddings": embeddings,
    "translation": translation,
}
save_path = "embeddings/embed-ezpz.pth"
torch.save(save, save_path)
print("saved to:", save_path)
