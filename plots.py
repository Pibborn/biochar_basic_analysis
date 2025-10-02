# this is chatgpt

def plot_feature_label_correlations(
    df,
    label: str,
    features=None,
    top_k: int | None = None,
    method: str = "pearson",
    dropna: str = "pairwise",
    show_scatter: bool = False,
    col_wrap: int = 4,
):
    """
    Plot linear correlations between numeric features and a single label.

    - method: 'pearson' (default). (Spearman also works for monotonic.)
    - dropna:
        * 'pairwise' -> compute each corr on rows valid for that pair (robust to sparse NaNs)
        * 'any'      -> drop any row with NaN in (features ∪ {label}) once, then compute
    - show_scatter: if True, draws facet scatterplots with regression lines.
    Returns a pd.Series of correlations sorted by |r| desc.
    """
    import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt

    if label not in df.columns:
        raise ValueError(f"label '{label}' not in DataFrame")

    num = df.select_dtypes(include=np.number)
    if label not in num.columns:
        raise ValueError(f"label '{label}' must be numeric")

    if features is None:
        features = [c for c in num.columns if c != label]
    else:
        features = [c for c in features if c in num.columns and c != label]

    if not features:
        raise ValueError("no numeric features to correlate")

    if dropna == "any":
        data = df[features + [label]].dropna()
        corr = data[features].corrwith(data[label], method=method)
    else:
        vals = {}
        y = df[label]
        for f in features:
            xy = pd.concat([df[f], y], axis=1).dropna()
            if len(xy) < 2 or xy[f].std(ddof=0) == 0 or xy[label].std(ddof=0) == 0:
                vals[f] = np.nan
            else:
                vals[f] = xy[f].corr(xy[label], method=method)
        corr = pd.Series(vals)

    corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
    if top_k is not None:
        corr = corr.iloc[:top_k]

    # Bar plot of correlations
    plt.figure(figsize=(8, max(4, 0.35 * len(corr))))
    sns.barplot(x=corr.values, y=corr.index, orient="h")
    plt.xlabel(f"{method.title()} r with {label}")
    plt.ylabel("Feature")
    plt.title(f"Feature ↔ {label} correlations")
    plt.tight_layout()
    plt.savefig('test_{}.png'.format(label.replace('/', '_')))

    # Optional scatter facets with best-fit lines
    if show_scatter and len(corr) > 0:
        m = df[corr.index.tolist() + [label]].dropna()
        long = m.melt(id_vars=label, var_name="feature", value_name="x")
        g = sns.lmplot(
            data=long, x="x", y=label, col="feature", col_wrap=col_wrap,
            scatter_kws={"alpha": 0.6}, line_kws={"linewidth": 2}
        )
        g.set_titles("{col_name}")
        g.set_xlabels("")
        g.set_ylabels(label)
        plt.tight_layout()
        plt.show()

    return corr

