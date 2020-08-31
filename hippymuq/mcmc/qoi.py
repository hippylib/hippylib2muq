import hippylib as hl


def cal_qoiTracer(pde, qoi, muq_samps):
    samps_mat = muq_samps.AsMatrix()
    nums = samps_mat.shape[1]
    tracer = hl.QoiTracer(nums)

    ct = 0
    u = pde.generate_state()
    m = pde.generate_parameter()
    while ct < nums:
        m.set_local(samps_mat[:,ct])
        x = [u, m, None]
        pde.solveFwd(u, x)
        q = qoi.eval([u, m])
        tracer.append(ct, q)
        ct += 1
    return tracer


def track_qoiTracer(pde, qoi, method_list, max_lag=None):
    qoi_dataset = dict()
    for mName, method in method_list.items():
        qoi_data = dict()
        samps = method['Samples']

        # Compute QOI
        tracer = cal_qoiTracer(pde, qoi, samps)

        # Estimate IAT
        iact, lags, acorrs = hl.integratedAutocorrelationTime(tracer.data, max_lag=max_lag)

        # Estimate ESS
        ess = samps.size() / iact

        # Save computed results
        qoi_data['qoi'] = tracer.data
        qoi_data['iact'] = iact
        qoi_data['ess'] = ess

        qoi_dataset[mName] = qoi_data
    return qoi_dataset
