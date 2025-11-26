def vit_block_flops(L, D, mlp_ratio=4):
    # QKV projections: 3 * (2 * L * D * D)
    flops_qkv = 3 * 2 * L * D * D
    
    # attention score: QK^T → (L, L)
    flops_qk = 2 * L * L * D
    
    # attention * V
    flops_av = 2 * L * L * D
    
    # output projection
    flops_proj = 2 * L * D * D
    
    # MLP: D → rD → D
    flops_mlp = 4 * L * mlp_ratio * D * D
    
    return flops_qkv + flops_qk + flops_av + flops_proj + flops_mlp

def estimate_flops(
    K, 
    D=768, 
    num_prefix=5, 
    num_patches_total=121, 
    mlp_ratio=4,
    encoder_layers=3,
    decoder_layers=1,
    img_size=154, 
    patch_size=14,
):
    
    # -------------------------------
    # 1. Resize FLOPs
    # -------------------------------
    F_resize = (img_size * img_size * 3 * 7)  # bilinear approx.

    # -------------------------------
    # 2. Patch embedding FLOPs (Conv2D)
    # -------------------------------
    Hout = img_size // patch_size    # = 154 // 14 = 11
    Wout = img_size // patch_size    # = 11
    Cin  = 3
    Cout = D
    k = patch_size

    F_patch = 2 * Cout * Hout * Wout * (Cin * k * k)

    # -------------------------------
    # 3. Encoder FLOPs (3 blocks, L = prefix + K)
    # -------------------------------
    L_enc = num_prefix + K
    F_enc_each = vit_block_flops(L_enc, D, mlp_ratio)
    F_enc_total = encoder_layers * F_enc_each

    # -------------------------------
    # 4. Decoder attention block (1 block)
    #    L = prefix + full 121 patches
    # -------------------------------
    L_dec = num_prefix + num_patches_total
    F_dec = decoder_layers * vit_block_flops(L_dec, D, mlp_ratio)

    # -------------------------------
    # 5. Linear head: 768 → 16 on 121 patches
    # -------------------------------
    F_head = 2 * num_patches_total * D * 16

    # -------------------------------
    # Total FLOPs
    # -------------------------------
    F_total = F_resize + F_patch + F_enc_total + F_dec + F_head

    # convert to GFLOPs
    return F_total / 1e9

total_flops = 0
for k in range(1, 11):
    total_flops += estimate_flops(K=k*12)

print(total_flops)

print(estimate_flops(K=121)*10)
