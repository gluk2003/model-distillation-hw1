#!/usr/bin/env python3
"""Offline BERT resource estimates. Standard library only; no inference or timing.

One multiply-accumulate is counted as two FLOPs. Matrix products only;
embeddings, bias additions, activations, softmax and LayerNorm FLOPs excluded.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIB = 2**20


def parameter_counts(c, labels=2):
    d, f, layers = c['hidden_size'], c['intermediate_size'], c['num_hidden_layers']
    return {
        'embeddings': (c['vocab_size'] + c['max_position_embeddings']
                       + c['type_vocab_size']) * d + 2 * d,
        'attention_projections': layers * 4 * (d * d + d),
        'ffn': layers * (d * f + f + f * d + d),
        'encoder_layer_norms': layers * 4 * d,
        'pooler': d * d + d,
        'classifier': d * labels + labels,
    }


def operations(c, batch, length, layers=None, ffn_dim=None):
    d = c['hidden_size']
    f = c['intermediate_size'] if ffn_dim is None else ffn_dim
    n = c['num_hidden_layers'] if layers is None else layers
    parts = {
        'projections_flops': n * 8 * batch * length * d**2,
        'ffn_flops': n * 4 * batch * length * d * f,
        'attention_products_flops': n * 4 * batch * length**2 * d,
    }
    total = sum(parts.values())
    return {
        'batch': batch, 'length': length, 'layers': n, 'ffn_dim': f,
        **parts,
        'encoder_matmul_flops': total,
        'gflops': total / 1e9,
        'ffn_percent': 100 * parts['ffn_flops'] / total,
        'attention_products_percent': 100 * parts['attention_products_flops'] / total,
        'one_attention_tensor_fp32_mib': batch * c['num_attention_heads'] * length**2 * 4 / MIB,
        'one_ffn_intermediate_fp32_mib': batch * length * f * 4 / MIB,
    }


def analyze():
    c = json.loads((ROOT / 'sources/bert_config.json').read_text())
    provenance = json.loads((ROOT / 'sources/provenance.json').read_text())
    counts = parameter_counts(c, provenance['num_labels_assumption'])
    total = sum(counts.values())
    # Quantize matrix weights in Linear layers only. Biases, embeddings and LN
    # stay FP32. Packed buffers, scales and zero-points are not included.
    d, f, n = c['hidden_size'], c['intermediate_size'], c['num_hidden_layers']
    linear_matrix_weights = n * (4 * d**2 + 2 * d * f) + d**2 + d * 2
    rows = [operations(c, 1, s) for s in (32, 64, 128, 256, 512)]
    base = operations(c, 1, 128)['encoder_matmul_flops']
    candidates = [
        ('BERT, S=128', operations(c, 1, 128)),
        ('BERT, S=64', operations(c, 1, 64)),
        ('6-layer BERT-shaped student, S=128', operations(c, 1, 128, layers=6)),
        ('BERT, FFN=2048, S=128', operations(c, 1, 128, ffn_dim=2048)),
    ]
    return {
        'kind': 'analytical_estimates_not_measurements',
        'model_id': provenance['model_id'], 'revision': provenance['revision'],
        'convention': '1 MAC = 2 FLOPs; encoder matrix multiplications only',
        'parameter_counts': counts, 'total_parameters': total,
        'fp32_weight_mib': total * 4 / MIB,
        'linear_matrix_weights': linear_matrix_weights,
        'linear_int8_other_fp32_weight_mib': (linear_matrix_weights + (total - linear_matrix_weights) * 4) / MIB,
        'linear_int8_excludes': ['scales', 'zero_points', 'packed_buffers', 'runtime_memory'],
        'by_sequence_length': rows,
        'candidates': [dict(name=name, **row,
                            flop_ratio_to_baseline=row['encoder_matmul_flops'] / base)
                       for name, row in candidates],
        'six_channel_ddr4_2933_theoretical_gb_s': 6 * 2933e6 * 8 / 1e9,
        'scenario': {
            'requests_per_second': 20, 'burst_requests_per_second': 40,
            'workers': 2, 'cores_per_worker': 4,
            'service_time_upper_bound_for_stability_at_40_rps_ms': 2 / 40 * 1000,
            'daily_requests_at_continuous_20_rps': 20 * 86400,
            'dense_encoder_weights_fp32_mib': (counts['attention_projections'] + counts['ffn']) * 4 / MIB,
            'dense_weight_streaming_ms_at_assumed_30_gb_s': (counts['attention_projections'] + counts['ffn']) * 4 / 30e9 * 1000,
        },
    }


def main():
    result = analyze()
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    (out / 'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = [
        '# Расчётные оценки BERT', '',
        'Сгенерировано `python3 scripts/analyze_bert.py`. Это не результаты профилирования.', '',
        f"Параметры: **{result['total_parameters']:,}**; FP32: **{result['fp32_weight_mib']:.2f} MiB**.",
        f"Матрицы Linear в INT8, остальное в FP32: **{result['linear_int8_other_fp32_weight_mib']:.2f} MiB** без служебных данных.", '',
        '| B | S | Encoder GFLOPs | FFN, % FLOPs | QKᵀ + AV, % FLOPs | Один attention tensor FP32, MiB | Один FFN intermediate FP32, MiB |',
        '|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in result['by_sequence_length']:
        lines.append(f"| {row['batch']} | {row['length']} | {row['gflops']:.3f} | {row['ffn_percent']:.2f} | {row['attention_products_percent']:.2f} | {row['one_attention_tensor_fp32_mib']:.2f} | {row['one_ffn_intermediate_fp32_mib']:.2f} |")
    lines += ['', '| Вариант | Encoder GFLOPs | Доля FLOPs от BERT S=128 |', '|---|---:|---:|']
    for row in result['candidates']:
        lines.append(f"| {row['name']} | {row['gflops']:.3f} | {row['flop_ratio_to_baseline']:.3f} |")
    (out / 'tables.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
