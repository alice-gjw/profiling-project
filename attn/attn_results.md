### Running the code
1. Choose whether you want to run Naive (`--eager`) or Flash Attention (`--flash_attn_2`)
2. Choose between the full kernel (`--full`) or just an attention layer (`--attn_layer`)

### Options
`python attn/profile_attn.py --eager --full`
`python attn/profile_attn.py --eager --attn_layer`
`python attn/profile_attn.py --flash_attn_2 --full`
`python attn/profile_attn.py --flash_attn_2 --attn_layer`