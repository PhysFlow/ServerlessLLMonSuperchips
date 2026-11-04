import argparse
import time

import pynvml as nvml

LINK_METRICS = (
    ("pcie_rx_mib_s", "NVML_GPM_METRIC_PCIE_RX_PER_SEC"),
    ("pcie_tx_mib_s", "NVML_GPM_METRIC_PCIE_TX_PER_SEC"),
    ("c2c_rx_mib_s", "NVML_GPM_METRIC_C2C_DATA_RX_PER_SEC"),
    ("c2c_tx_mib_s", "NVML_GPM_METRIC_C2C_DATA_TX_PER_SEC"),
)


def read_metrics(samples, metric_ids):
    request = nvml.c_nvmlGpmMetricsGet_t()
    request.version = nvml.NVML_GPM_METRICS_GET_VERSION
    request.numMetrics = len(metric_ids)
    request.sample1, request.sample2 = samples
    for index, metric_id in enumerate(metric_ids):
        request.metrics[index].metricId = metric_id
    nvml.nvmlGpmMetricsGet(request)
    return [
        f"{request.metrics[index].value:.2f}" if request.metrics[index].nvmlReturn == 0 else ""
        for index in range(len(metric_ids))
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mig-csv", required=True)
    parser.add_argument("--link-csv", required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("uuids", nargs="+")
    args = parser.parse_args()

    nvml.nvmlInit()
    link_names = [name for name, attribute in LINK_METRICS if hasattr(nvml, attribute)]
    link_ids = [getattr(nvml, attribute) for _, attribute in LINK_METRICS if hasattr(nvml, attribute)]

    migs, parents = [], {}
    for uuid in args.uuids:
        handle = nvml.nvmlDeviceGetHandleByUUID(uuid)
        parent = nvml.nvmlDeviceGetDeviceHandleFromMigDeviceHandle(handle)
        instance = nvml.nvmlDeviceGetGpuInstanceId(handle)
        samples = [nvml.nvmlGpmSampleAlloc(), nvml.nvmlGpmSampleAlloc()]
        nvml.nvmlGpmMigSampleGet(parent, instance, samples[0])
        migs.append((uuid, handle, parent, instance, samples))
        parent_uuid = nvml.nvmlDeviceGetUUID(parent)
        parent_uuid = parent_uuid.decode() if isinstance(parent_uuid, bytes) else parent_uuid
        if parent_uuid not in parents:
            parent_samples = [nvml.nvmlGpmSampleAlloc(), nvml.nvmlGpmSampleAlloc()]
            nvml.nvmlGpmSampleGet(parent, parent_samples[0])
            parents[parent_uuid] = (parent, parent_samples)

    with open(args.mig_csv, "w", buffering=1) as mig_file, open(args.link_csv, "w", buffering=1) as link_file:
        mig_file.write("time,mig_uuid,dram_bw_util_percent,memory_used_mib\n")
        link_file.write("time,gpu_uuid," + ",".join(link_names) + "\n")
        while True:
            time.sleep(args.interval)
            now = time.strftime("%Y-%m-%dT%H:%M:%S")
            for uuid, handle, parent, instance, samples in migs:
                nvml.nvmlGpmMigSampleGet(parent, instance, samples[1])
                dram = read_metrics(samples, [nvml.NVML_GPM_METRIC_DRAM_BW_UTIL])[0]
                memory_used_mib = nvml.nvmlDeviceGetMemoryInfo(handle).used // 2**20
                mig_file.write(f"{now},{uuid},{dram},{memory_used_mib}\n")
                samples.reverse()
            for parent_uuid, (parent, samples) in parents.items():
                nvml.nvmlGpmSampleGet(parent, samples[1])
                link_file.write(f"{now},{parent_uuid}," + ",".join(read_metrics(samples, link_ids)) + "\n")
                samples.reverse()


if __name__ == "__main__":
    main()
