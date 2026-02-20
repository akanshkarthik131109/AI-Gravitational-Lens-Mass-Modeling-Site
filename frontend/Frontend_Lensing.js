// import React, { useState } from 'react';

// function App() {
//   const [file, setFile] = useState(null);
//   const [result, setResult] = useState(null);
//   const [loading, setLoading] = useState(false);

//   const handleUpload = async () => {
//     if (!file) return;
//     setLoading(true);

//     const formData = new FormData();
//     formData.append('file', file);

//     try {
//       const response = await fetch('http://127.0.0.1:8000/predict', {
//         method: 'POST',
//         body: formData,
//       });
//       const data = await response.json();
//       setResult(data);
//     } catch (error) {
//       console.error("Error uploading:", error);
//     } finally {
//       setLoading(false);
//     }
//   };

//   return (
//     <div className="p-10 bg-gray-100 min-h-screen">
//       <h1 className="text-3xl font-bold mb-6">Gravitational Lensing Analyzer</h1>
      
//       <div className="bg-white p-6 rounded-lg shadow-md max-w-md">
//         <input type="file" onChange={(e) => setFile(e.target.files[0])} className="mb-4" />
//         <button 
//           onClick={handleUpload}
//           className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 w-full"
//           disabled={loading}
//         >
//           {loading ? 'Analyzing...' : 'Run Prediction'}
//         </button>
//       </div>

//       {result && (
//         <div className="mt-10 grid grid-cols-1 md:grid-cols-2 gap-6">
//           <div className="bg-white p-6 rounded-lg shadow-md">
//             <h2 className="text-xl font-bold mb-4">Lensing Parameters</h2>
//             <ul className="space-y-2">
//               <li><strong>Theta_E:</strong> {result["SIE Info"][0].toFixed(4)}</li>
//               <li><strong>e1:</strong> {result["SIE Info"][1].toFixed(4)}</li>
//               <li><strong>e2:</strong> {result["SIE Info"][2].toFixed(4)}</li>
//               <li><strong>Center:</strong> ({result["SIE Info"][3]}, {result["SIE Info"][4]})</li>
//             </ul>
//           </div>
//         </div>
//       )}
//     </div>
//   );
// }

// export default App;
