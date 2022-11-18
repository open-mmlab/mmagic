# key-in-metafile: key-in-results.pkl
METRICS_MAPPING = {
    'FID': {
        'keys': ['FID-Full-50k/fid'],
        'tolerance': 0.5,
        'rule': 'less'
    },
    'PSNR': {
        'keys': ['PSNR'],
        'tolerance': 0.1,
        'rule': 'larger'
    },
    'SSIM': {
        'keys': ['MattingSSIM', 'SSIM'],
        'tolerance': 0.1,
        'rule': 'larger'
    },
    'l1 error': {
        'keys': ['MAE'],
        'tolerance': 0.1,
        'rule': 'less'
    },
    'CONN': {
        'keys': ['ConnectivityError'],
        'tolerance': 0.1,
        'rule': 'less'
    },
    'GRAD': {
        'keys': ['GradientError'],
        'tolerance': 0.1,
        'rule': 'less'
    },
    'MSE': {
        'keys': ['MSE', 'MattingMSE'],
        'tolerance': 0.1,
        'rule': 'less'
    },
    'SAD': {
        'keys': ['SAD'],
        'tolerance': 0.1,
        'rule': 'less'
    },
    'NIQE (Y)': {
        'keys': ['NIQE'],
        'tolerance': 0.1,
        'rule': 'less'
    },
    # verbose metric mapping for VSR
    'REDS4 (BIx4) PSNR (RGB)': {
        'keys': ['REDS4-BIx4-RGB/PSNR'],
        'tolerance': 0.1,
        'rule': 'larger'
    },
    'Vimeo-90K-T (BIx4) PSNR (Y)': {
        'keys': ['Vimeo-90K-T-BIx4-Y/PSNR'],
        'tolerance': 0.1,
        'rule': 'larger'
    },
    'Vimeo-90K-T (BDx4) PSNR (Y)': {
        'keys': ['Vimeo-90K-T-BDx4-Y/PSNR'],
        'tolerance': 0.1,
        'rule': 'larger'
    },
    'UDM10 (BDx4) PSNR (Y)': {
        'keys': ['UDM10-BDx4-Y/PSNR'],
        'tolerance': 0.1,
        'rule': 'larger'
    },
    'Vid4 (BDx4) PSNR (Y)': {
        'keys': ['VID4-BDx4-Y/PSNR'],
        'tolerance': 0.1,
        'rule': 'larger'
    },
    'Vid4 (BIx4) PSNR (Y)': {
        'keys': ['VID4-BIx4-Y/PSNR'],
        'tolerance': 0.1,
        'rule': 'larger'
    },
    'SPMCS-30 (BDx4) PSNR (Y)': {
        'keys': ['SPMCS-BDx4-Y/PSNR'],
        'tolerance': 0.1,
        'rule': 'larger'
    },
    'SPMCS-30 (BIx4) PSNR (Y)': {
        'keys': ['SPMCS-BIx4-Y/PSNR'],
        'tolerance': 0.1,
        'rule': 'larger'
    },
    'REDS4 (BIx4) SSIM (RGB)': {
        'keys': ['REDS4-BIx4-RGB/SSIM'],
        'tolerance': 0.1,
        'rule': 'larger'
    },
    'Vimeo-90K-T (BIx4) SSIM (Y)': {
        'keys': ['Vimeo-90K-T-BIx4-Y/SSIM'],
        'tolerance': 0.1,
        'rule': 'larger'
    },
    'Vimeo-90K-T (BDx4) SSIM (Y)': {
        'keys': ['Vimeo-90K-T-BDx4-Y/SSIM'],
        'tolerance': 0.1,
        'rule': 'larger'
    },
    'UDM10 (BDx4) SSIM (Y)': {
        'keys': ['UDM10-BDx4-Y/SSIM'],
        'tolerance': 0.1,
        'rule': 'larger'
    },
    'Vid4 (BDx4) SSIM (Y)': {
        'keys': ['VID4-BDx4-Y/SSIM'],
        'tolerance': 0.1,
        'rule': 'larger'
    },
    'Vid4 (BIx4) SSIM (Y)': {
        'keys': ['VID4-BIx4-Y/SSIM'],
        'tolerance': 0.1,
        'rule': 'larger'
    },
    'SPMCS-30 (BDx4) SSIM (Y)': {
        'keys': ['SPMCS-BDx4-Y/SSIM'],
        'tolerance': 0.1,
        'rule': 'larger'
    },
    'SPMCS-30 (BIx4) SSIM (Y)': {
        'keys': ['SPMCS-BIx4-Y/SSIM'],
        'tolerance': 0.1,
        'rule': 'larger'
    },
    'Set5 SSIM': {
        'keys': ['Set5/SSIM'],
        'tolerance': 0.1,
        'rule': 'larger'
    },
    'Set5 PSNR': {
        'keys': ['Set5/PSNR'],
        'tolerance': 0.1,
        'rule': 'larger'
    },
    'Set14 SSIM': {
        'keys': ['Set14/SSIM'],
        'tolerance': 0.1,
        'rule': 'larger'
    },
    'Set14 PSNR': {
        'keys': ['Set14/PSNR'],
        'tolerance': 0.1,
        'rule': 'larger'
    },
    'DIV2K SSIM': {
        'keys': ['DIV2K/SSIM'],
        'tolerance': 0.1,
        'rule': 'larger'
    },
    'DIV2K PSNR': {
        'keys': ['DIV2K/PSNR'],
        'tolerance': 0.1,
        'rule': 'larger'
    },
}

# add LIIF metrics
liif_metrics = {}
for dataset in ['Set5', 'Set14', 'DIV2K']:
    for metric, tolerance in zip(['PSNR', 'SSIM'], [0.1, 0.1]):
        for scale in [2, 3, 4, 6, 18, 30]:
            liif_metrics[f'{dataset}x{scale} {metric}'] = {
                'keys': [f'{dataset}x{scale}/{metric}'],
                'tolerance': 0.1,
                'rule': 'larger'
            }
METRICS_MAPPING.update(liif_metrics)


def filter_metric(metric_mapping, summary_data):

    used_metric = dict()
    for metric, metric_info in metric_mapping.items():
        for data_dict in summary_data.values():
            if metric in data_dict:
                used_metric[metric] = metric_info
                break

    return used_metric
