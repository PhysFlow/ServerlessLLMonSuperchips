def link_bytes_per_s(nvml, gpu) -> float:
    count_field = nvml.nvmlDeviceGetFieldValues(
        gpu, [nvml.NVML_FI_DEV_C2C_LINK_COUNT]
    )[0]
    if count_field.nvmlReturn == 0 and count_field.value.uiVal > 0:
        count = count_field.value.uiVal
        fields = nvml.nvmlDeviceGetFieldValues(
            gpu,
            [(nvml.NVML_FI_DEV_C2C_LINK_GET_MAX_BW, index) for index in range(count)],
        )
        speeds = [field.value.uiVal for field in fields if field.nvmlReturn == 0]
        if len(speeds) != count or any(speed <= 0 for speed in speeds):
            raise RuntimeError("NVML did not report every active C2C link bandwidth")
        return sum(speeds) * 1e6

    lane_mbps = nvml.nvmlDeviceGetPcieSpeed(gpu)
    lanes = nvml.nvmlDeviceGetCurrPcieLinkWidth(gpu)
    if lane_mbps <= 0 or lanes <= 0:
        raise RuntimeError("NVML did not report a valid PCIe link bandwidth")
    return lane_mbps * 1e6 / 8 * lanes * 128 / 130
