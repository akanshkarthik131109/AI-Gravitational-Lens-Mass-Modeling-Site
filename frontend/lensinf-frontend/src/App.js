import React, { useState } from 'react';


function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [pixelScale, setPixelScale] = useState(0.131); // Default value
  const [lensZ, setLensZ] = useState(0.1371); // Default value
  const [sourceZ, setSourceZ] = useState(0.7126); // Default value



  const impath_1 = "public/Test_Pics/lens-141-_jpg.rf.f2d971e2e122c0037c9211d195ab1d83.jpg"
  const impath_2 = "public/Test_Pics/lens-156-_jpg.rf.02900da393dbc901560fc92c2dbb8376.jpg"
  const impath_3 = "public/Test_Pics/lens-166-_jpg.rf.b8c0b20d1ea024aca904c9e800922949.jpg"
  const impath_4 = "public/Test_Pics/lens-198-_jpg.rf.7af93cbf1ad2bf848ff5a1b6d1ca8f69.jpg"
  const impath_5 = "public/Test_Pics/lens-200-_jpg.rf.88a0edc79a9afd20f9efb6a62ef65195.jpg"

  const imName_1 = "lens-141-_jpg.rf.f2d971e2e122c0037c9211d195ab1d83.jpg"
  const imName_2 = "lens-156-_jpg.rf.02900da393dbc901560fc92c2dbb8376.jpg"
  const imName_3 = "lens-166-_jpg.rf.b8c0b20d1ea024aca904c9e800922949.jpg"
  const imName_4 = "lens-198-_jpg.rf.7af93cbf1ad2bf848ff5a1b6d1ca8f69.jpg"
  const imName_5 = "lens-200-_jpg.rf.88a0edc79a9afd20f9efb6a62ef65195.jpg"



  // Function to load a file from URL
  const loadFileFromUrl = async (url, filename) => {
    try {
      const response = await fetch(url);
      const blob = await response.blob();
      const file = new File([blob], filename, { type: blob.type });
      setFile(file);
    } catch (error) {
      console.error("Error loading file:", error);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('pixel_scale', pixelScale); 
    formData.append('lens_z', lensZ);    // Add this
    formData.append('source_z', sourceZ); // Add this

    try {
      const response = await fetch('http://18.220.50.214/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Error uploading:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    
    <div className="p-10 min-h-screen" style={{
        backgroundImage: 'url(https://scitechdaily.com/images/Galaxy-Cluster-Abell-370-Hubble-Rotated-scaled.jpg)',
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        backgroundRepeat: 'no-repeat'
      }}>
      <h1 h1 className = "text-white" style={{ textAlign: 'center', fontSize: '60px' }}><strong>Weighing the Cosmos</strong></h1>
      <div className="flex gap-4 justify-center mt-8 mb-6">
        <button className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
          onClick={() => loadFileFromUrl(impath_1, imName_1)}>
          Pre-prepped Lens #1
        </button>
        <button className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700" 
        onClick={() => loadFileFromUrl(impath_2, imName_2)}>
          Pre-prepped Lens #2
        </button>
        <button className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
        onClick={() => loadFileFromUrl(impath_3, imName_3)}>
          Pre-prepped Lens #3
        </button>
        <button className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
        onClick={() => loadFileFromUrl(impath_4, imName_4)}>
          Pre-prepped Lens #4
        </button>
        <button className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
          onClick={() => loadFileFromUrl(impath_5, imName_5)}>
          Pre-prepped Lens #5
        </button>
      </div>
      <div className="bg-white p-6 rounded-lg shadow-md max-w-4xl mx-auto mt-8">
        <div className="flex gap-4 items-center flex flex-row gap-8 items-center justify-center w-full my-10">
              <div className="flex flex-col gap-4 max-w-xs">
                <label className="text-sm text-xl font-semibold text-gray-700 mb-1 shrink-0">
                  Pixel Scale (arcsec/pixel) 
                </label>
              
                <input 
                  type="number" 
                  step="0.001"
                  value={pixelScale}
                  onChange={(e) => setPixelScale(parseFloat(e.target.value))}
                  className="p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 outline-none shrink-0 text-sm"
                />

                <label className="text-sm text-xl font-semibold text-gray-700 mb-1 shrink-0">
                  Lens Redshift  
                </label>
              
                <input 
                  type="number" 
                  step="0.001"
                  value={lensZ}
                  onChange={(e) => setLensZ(parseFloat(e.target.value))}
                  className="p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 outline-none shrink-0 text-sm"
                />

                <label className="text-sm text-xl font-semibold text-gray-700 mb-1 shrink-0">
                  Source Redshift
                </label>
              
                <input 
                  type="number" 
                  step="0.001"
                  value={sourceZ}
                  onChange={(e) => setSourceZ(parseFloat(e.target.value))}
                  className="p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 outline-none shrink-0 text-sm"
                />

              </div>
          
            <input type="file" onChange={(e) => setFile(e.target.files[0])} className="mb-4" />
          
          
        </div>
        <button 
          onClick={handleUpload}
          className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 w-full"
          disabled={loading}
        >
          {loading ? 'Analyzing...' : 'Run Prediction'}
        </button>



      </div>

      

      {result && (
        <div>
          <div className="bg-white p-6 rounded-lg shadow-md gap-8 justify-center w-full mt-20">
            <h2 className="text-xl font-bold mb-4">Lensing Parameters</h2>
            <ul className="space-y-2">
              <li><strong>Theta_E:</strong> {result["SIE_Info"][0].toFixed(4)}</li>
              <li><strong>e1:</strong> {result["SIE_Info"][1].toFixed(4)}</li>
              <li><strong>e2:</strong> {result["SIE_Info"][2].toFixed(4)}</li>
              <li><strong>Center:</strong> ({result["SIE_Info"][3]}, {result["SIE_Info"][4]})</li>
              <li><strong>Mass(In Solar Masses):</strong> {result["Mass"]}</li>
            </ul>
          </div>
        <div className="mt-10 grid grid-cols-1 md:grid-cols-2 gap-6">
          
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-bold mb-4">Image</h2>
            <img 
              src={`data:image/png;base64,${result.Image}`} // This matches the backend key "Image"
              alt="Original with Mask"
              className="w-full h-auto rounded border"
            />
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-bold mb-4">Arc Mask</h2>
            <img 
              src={`data:image/png;base64,${result.overlay}`} // This matches the backend key "Image"
              alt="Original with Mask"
              className="w-full h-auto rounded border"
            />
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-bold mb-4">Ellipse</h2>
            <img 
              src={`data:image/png;base64,${result.ellipse}`} // This matches the backend key "Image"
              alt="Original with Mask"
              className="w-full h-auto rounded border"
            />
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-bold mb-4">Convergence Map</h2>
            <img 
              src={`data:image/png;base64,${result.convergence_map}`}
              alt="Convergence Map"
              className="w-full h-auto rounded"
            />
          </div>

        </div>
        </div>
      )}
    </div>
  );
}

export default App;
