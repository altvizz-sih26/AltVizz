function Home() {
  return <Layout>
    <section className="hero">
      <p className="eyebrow">SATELLITE TERRAIN INTELLIGENCE</p>
      <h1>From satellite imagery<br />to <i>3D terrain.</i></h1>
      <p className="lede">Transform a single optical image into elevation-aware terrain and an interactive 3D model.</p>
      <div className="actions">
        <Link className="button primary" to="/upload">Start Reconstruction <Arrow /></Link>
        <a className="button secondary" href="#workflow">Explore Workflow</a>
      </div>
    </section>


    {/* NEW: choose-your-path cards */}
    <section className="section">
      <p className="eyebrow">CHOOSE YOUR PATH</p>
      <h2>Two ways to reconstruct terrain.</h2>
      <div className="mode-grid">
        <Link className="mode-card" to="/upload">
          <span className="mode-icon">◭</span>
          <p className="eyebrow">FULL PIPELINE</p>
          <h3>Terrain Analysis Suite</h3>
          <p>Upload a single image, run depth estimation, calibrate with SRTM reference elevation, and inspect the DSM and 3D mesh.</p>
          <b>Start analysis <Arrow /></b>
        </Link>
        <a className="mode-card" href="/viewer.html">
          <span className="mode-icon">◒</span>
          <p className="eyebrow">COMPARATIVE ANALYSIS</p>
          <h3>Before / After Terrain Compare</h3>
          <p>Upload two images of the same site and compare elevation, DSM and 3D terrain side by side over time.</p>
          <b>Launch viewer <Arrow /></b>
        </a>
      </div>
    </section>

    
    <section id="workflow" className="section">
      <p className="eyebrow">THE RECONSTRUCTION PATH</p>
      <h2>Terrain intelligence, layer by layer.</h2>
      {/* CHANGED: workflow steps now sit inside a card instead of loose on the background */}
      <div className="workflow-card">
        <div className="workflow">
          {['Satellite Image', 'Depth Estimation', 'Terrain Analysis', 'Elevation Calibration', 'DSM', '3D Terrain'].map((x, i) =>
            <div className="flow" key={x}>
              <div className="flow-num">0{i + 1}</div>
              <span>{x}</span>
              {i < 5 && <Arrow />}
            </div>
          )}
        </div>
      </div>
    </section>

    

    <section className="split section">
      <div>
        <p className="eyebrow">FROM 2D TO TOPOGRAPHY</p>
        <h2>Seeing height where imagery sees colour.</h2>
      </div>
      <p>DepthWizard brings together monocular depth estimation, terrain-aware calibration and reference elevation data to create a meaningful 3D terrain interpretation—without overstating certainty.</p>
    </section>
  </Layout>;
}
