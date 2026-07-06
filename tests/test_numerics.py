"""Tests for the acceleration layer: backends, batched kernels, blocking, and workers."""

import numpy as np
import pytest

from quantum_transport import (
    AndersonImpurity,
    LeadSelfEnergy,
    RashbaRingDevice,
    available_backends,
    backend_name,
    batched_retarded_green,
    batched_transmission,
    get_backend,
    parallel_map,
    sigma_stack,
    to_numpy,
)
from quantum_transport.numerics import blocked_over_grid


class TestBackendResolution:
    def test_numpy_backend(self):
        assert get_backend("numpy") is np
        assert get_backend("cpu") is np
        assert backend_name(get_backend("numpy")) == "numpy"

    def test_auto_backend_resolves(self):
        xp = get_backend("auto")
        assert backend_name(xp) in {"numpy", "cupy"}

    def test_none_means_auto(self):
        assert backend_name(get_backend(None)) in {"numpy", "cupy"}

    def test_module_passthrough(self):
        assert get_backend(np) is np

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError):
            get_backend("tensorflow")

    def test_cupy_backend_errors_cleanly_when_missing(self):
        report = available_backends()
        assert report["numpy"] is True
        if not report["cupy"]:
            with pytest.raises((ImportError, RuntimeError)):
                get_backend("cupy")

    def test_to_numpy_passthrough(self):
        arr = np.arange(3.0)
        assert to_numpy(arr) is arr


class TestParallelMap:
    def test_serial_and_threaded_agree(self):
        values = list(range(20))
        serial = parallel_map(lambda x: x * x, values)
        threaded = parallel_map(lambda x: x * x, values, workers=4)
        assert serial == threaded == [x * x for x in values]

    def test_preserves_order(self):
        result = parallel_map(lambda x: -x, range(50), workers=8)
        assert result == [-x for x in range(50)]


class TestBlockedOverGrid:
    def test_blocking_matches_single_shot(self):
        grid = np.linspace(-1, 1, 37)

        def kernel(subgrid):
            return subgrid[:, None, None] * np.ones((1, 2, 2))

        full = kernel(grid)
        blocked = blocked_over_grid(kernel, grid, dim=2, target_bytes=16 * 5 * 4)
        np.testing.assert_allclose(full, blocked)

    def test_workers_split_matches(self):
        grid = np.linspace(-1, 1, 41)

        def kernel(subgrid):
            return np.sin(subgrid)

        np.testing.assert_allclose(
            blocked_over_grid(kernel, grid, dim=1, workers=4),
            np.sin(grid),
        )


def _ring_view():
    device = RashbaRingDevice(n_sites=5, gamma=1.0, lambda_r=0.3, phi_over_phi0=0.15)
    dim = device.dim
    coupling_left = np.zeros((dim, dim))
    coupling_left[0, 0] = coupling_left[1, 1] = 0.8
    coupling_right = np.zeros((dim, dim))
    coupling_right[4, 4] = coupling_right[5, 5] = 0.8
    left = LeadSelfEnergy.semi_infinite_chain(coupling_left, hopping=2.0, mu=0.2, temperature=0.05)
    right = LeadSelfEnergy.semi_infinite_chain(coupling_right, hopping=2.0, mu=-0.2, temperature=0.05)
    return device.transport(left, right)


GRID = np.linspace(-1.5, 1.5, 61)
ETA = 1e-6


class TestMatrixTransportBatchedEquivalence:
    """The batched fast paths must reproduce the scalar per-frequency methods exactly."""

    def test_retarded_values(self):
        view = _ring_view()
        loop = np.array([view.retarded(float(w), eta=ETA) for w in GRID])
        np.testing.assert_allclose(view.retarded_values(GRID, eta=ETA), loop, atol=1e-12)

    def test_transmission_values(self):
        view = _ring_view()
        loop = np.array([view.transmission(float(w), eta=ETA) for w in GRID])
        np.testing.assert_allclose(view.transmission_values(GRID, eta=ETA), loop, atol=1e-12)

    def test_lesser_and_greater_values(self):
        view = _ring_view()
        lesser_loop = np.array([view.lesser(float(w), eta=ETA) for w in GRID])
        greater_loop = np.array([view.greater(float(w), eta=ETA) for w in GRID])
        np.testing.assert_allclose(view.lesser_values(GRID, eta=ETA), lesser_loop, atol=1e-12)
        np.testing.assert_allclose(view.greater_values(GRID, eta=ETA), greater_loop, atol=1e-12)

    def test_current_spectral_density_values(self):
        view = _ring_view()
        loop = np.array([view.current_spectral_density(float(w), lead="left", eta=ETA) for w in GRID])
        np.testing.assert_allclose(
            view.current_spectral_density_values(GRID, lead="left", eta=ETA), loop, atol=1e-12
        )

    def test_spin_resolved_paths(self):
        view = _ring_view()
        transmission_loop = np.array(
            [view.spin_resolved_transmission(float(w), "+", eta=ETA, axis="y") for w in GRID]
        )
        np.testing.assert_allclose(
            view.spin_resolved_transmission_values(GRID, "+", eta=ETA, axis="y"),
            transmission_loop,
            atol=1e-12,
        )
        current_loop = np.array(
            [
                view.spin_resolved_current_spectral_density(float(w), "+", lead="left", eta=ETA, axis="z")
                for w in GRID
            ]
        )
        np.testing.assert_allclose(
            view.spin_resolved_current_spectral_density_values(GRID, "+", lead="left", eta=ETA, axis="z"),
            current_loop,
            atol=1e-12,
        )

    def test_spin_polarization_values(self):
        view = _ring_view()
        loop = np.array([view.spin_polarization(float(w), eta=ETA, axis="z") for w in GRID])
        np.testing.assert_allclose(view.spin_polarization_values(GRID, eta=ETA, axis="z"), loop, atol=1e-10)

    def test_workers_agree_with_serial(self):
        view = _ring_view()
        serial = view.transmission_values(GRID, eta=ETA)
        threaded = view.transmission_values(GRID, eta=ETA, workers=4)
        np.testing.assert_allclose(threaded, serial, atol=1e-12)

    def test_wide_band_fast_path_matches_scalar(self):
        """Omega-independent leads take a broadcast fast path; results must not change."""
        device = RashbaRingDevice(n_sites=5, gamma=1.0, lambda_r=0.3, phi_over_phi0=0.15)
        dim = device.dim
        gl = np.zeros((dim, dim))
        gl[0, 0] = gl[1, 1] = 0.6
        gr = np.zeros((dim, dim))
        gr[4, 4] = gr[5, 5] = 0.4
        left = LeadSelfEnergy.wide_band(gl, mu=0.3, temperature=0.1)
        right = LeadSelfEnergy.wide_band(gr, mu=-0.3, temperature=0.1)
        assert left.omega_independent and right.omega_independent
        view = device.transport(left, right)
        for scalar, batched in (
            (view.transmission, view.transmission_values),
            (view.lesser, view.lesser_values),
            (view.greater, view.greater_values),
        ):
            loop = np.array([scalar(float(w), eta=ETA) for w in GRID])
            np.testing.assert_allclose(batched(GRID, eta=ETA), loop, atol=1e-12)
        current_loop = np.array(
            [view.current_spectral_density(float(w), lead="left", eta=ETA) for w in GRID]
        )
        np.testing.assert_allclose(
            view.current_spectral_density_values(GRID, lead="left", eta=ETA), current_loop, atol=1e-12
        )

    def test_single_precision_close_to_double(self):
        view = _ring_view()
        double = view.transmission_values(GRID, eta=1e-4)
        single = view.transmission_values(GRID, eta=1e-4, precision="single")
        np.testing.assert_allclose(single, double, rtol=5e-4, atol=5e-4)
        with pytest.raises(ValueError):
            view.transmission_values(GRID, eta=1e-4, precision="half")

    def test_kernels_directly(self):
        view = _ring_view()
        sig_l = sigma_stack(view.left_lead.sigma_retarded, GRID)
        sig_r = sigma_stack(view.right_lead.sigma_retarded, GRID)
        g_r = batched_retarded_green(view.hamiltonian, sig_l + sig_r, GRID, eta=ETA)
        gamma_l = np.stack([view.left_lead.gamma(float(w)) for w in GRID])
        gamma_r = np.stack([view.right_lead.gamma(float(w)) for w in GRID])
        transmission = batched_transmission(g_r, gamma_l, gamma_r)
        loop = np.array([view.transmission(float(w), eta=ETA) for w in GRID])
        np.testing.assert_allclose(transmission, loop, atol=1e-12)


class TestOpenAndersonBatchedEquivalence:
    def _view(self):
        return AndersonImpurity(eps=-0.5, U=2.0).open(
            {"up": 0.3, "down": 0.4},
            0.35,
            mu_left=0.2,
            mu_right=-0.2,
            temperature_left=0.05,
            temperature_right=0.05,
        )

    OCCUPATIONS = {"up": 0.42, "down": 0.58}
    WGRID = np.linspace(-6.0, 6.0, 101)

    def test_retarded_values(self):
        view = self._view()
        loop = np.array([view.retarded(float(w), eta=1e-3, occupations=self.OCCUPATIONS) for w in self.WGRID])
        np.testing.assert_allclose(
            view.retarded_values(self.WGRID, eta=1e-3, occupations=self.OCCUPATIONS), loop, atol=1e-12
        )

    def test_lesser_greater_transmission(self):
        view = self._view()
        for scalar, batched in (
            (view.lesser, view.lesser_values),
            (view.greater, view.greater_values),
        ):
            loop = np.array([scalar(float(w), eta=1e-3, occupations=self.OCCUPATIONS) for w in self.WGRID])
            np.testing.assert_allclose(
                batched(self.WGRID, eta=1e-3, occupations=self.OCCUPATIONS), loop, atol=1e-12
            )
        transmission_loop = np.array(
            [view.transmission(float(w), eta=1e-3, occupations=self.OCCUPATIONS) for w in self.WGRID]
        )
        np.testing.assert_allclose(
            view.transmission_values(self.WGRID, eta=1e-3, occupations=self.OCCUPATIONS),
            transmission_loop,
            atol=1e-12,
        )

    def test_meir_wingreen_current(self):
        view = self._view()
        loop = np.trapezoid(
            [
                view.meir_wingreen_current_density(float(w), lead="left", eta=1e-3, occupations=self.OCCUPATIONS)
                for w in self.WGRID
            ],
            self.WGRID,
        )
        batched = view.meir_wingreen_current(self.WGRID, lead="left", eta=1e-3, occupations=self.OCCUPATIONS)
        assert batched == pytest.approx(float(loop), rel=1e-10)

    def test_channel_occupation_and_scf(self):
        view = self._view()
        occupation = view.channel_occupation("up", self.WGRID, eta=1e-3, occupations=self.OCCUPATIONS)
        assert 0.0 <= occupation <= 1.0
        result = view.self_consistent_occupations(self.WGRID, eta=1e-3, workers=2, tol=1e-9)
        assert result.converged
        result_serial = view.self_consistent_occupations(self.WGRID, eta=1e-3, tol=1e-9)
        assert result.occupations["up"] == pytest.approx(result_serial.occupations["up"], abs=1e-9)
        assert result.occupations["down"] == pytest.approx(result_serial.occupations["down"], abs=1e-9)
