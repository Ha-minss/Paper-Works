from __future__ import annotations
from stabilitycheck.configs import EngineConfig, default_schema_did
from stabilitycheck.adapters.dispatch import AdapterRegistry
from stabilitycheck.adapters.did import DIDAdapter
from stabilitycheck.dgp import simulate_panel
from stabilitycheck.runner import run_paper_pipeline

def main():
    df = simulate_panel(seed=7)
    schema = default_schema_did(controls_ranked=["x1","x2","x3"])

    cfg = EngineConfig(outdir="paper_outputs", seed=7)
    cfg.anchor_main = schema.tiers["Tier2_Contested"].designs[0]
    cfg.anchor_cons = schema.tiers["Tier2_Contested"].designs[-1]
    cfg.anchor_alt  = schema.tiers["Tier2_Contested"].designs[1]

    reg = AdapterRegistry()
    reg.register("DID", DIDAdapter(schema))

    out = run_paper_pipeline(cfg, schema, reg, df)
    print("DONE. Outputs in ./paper_outputs/")
    print(out["tier_summary"].head())

if __name__ == "__main__":
    main()
