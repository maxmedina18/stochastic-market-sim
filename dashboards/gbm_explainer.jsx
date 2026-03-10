export default function GBMExplainer() {
  return (
    <section style={{ fontFamily: "Georgia, serif", maxWidth: 860, margin: "0 auto", padding: "2rem" }}>
      <h1>Geometric Brownian Motion</h1>
      <p>
        This dashboard is intentionally narrow: it explains the GBM process used in the research codebase and the
        analytical checks used to validate simulations.
      </p>
      <p>
        Model: dS_t = mu S_t dt + sigma S_t dW_t
      </p>
      <p>
        Closed form: S_t = S_0 exp((mu - 0.5 sigma^2)t + sigma W_t)
      </p>
      <ul>
        <li>Drift `mu` controls expected growth.</li>
        <li>Volatility `sigma` controls dispersion.</li>
        <li>Validation compares empirical terminal moments against analytical moments.</li>
      </ul>
    </section>
  );
}
