import { useState, useRef, useEffect } from "react";
import { ZoomIn, ZoomOut } from "lucide-react";

const segments = [
  { id: 0, name: "Segment 0", color: "#18181b", points: 63 }, // Noir
  { id: 1, name: "Segment 1", color: "#3f3f46", points: 48 }, // Gris foncé
  { id: 2, name: "Segment 2", color: "#71717a", points: 39 }, // Gris moyen
  { id: 3, name: "Segment 3", color: "#a1a1aa", points: 58 }, // Gris clair
  { id: 4, name: "Segment 4", color: "#525252", points: 34 }, // Gris neutre
  { id: 5, name: "Segment 5", color: "#57534e", points: 45 }, // Gris chaud
  { id: 6, name: "Segment 6", color: "#44403c", points: 41 }, // Gris pierre
  { id: 7, name: "Segment 7", color: "#292524", points: 74 }, // Noir chaud
];

// Generate mock 3D data points
const generatePoints = () => {
  const points: Array<{ x: number; y: number; z: number; segment: number; color: string }> = [];
  
  segments.forEach((segment) => {
    const centerX = (Math.random() - 0.5) * 300;
    const centerY = (Math.random() - 0.5) * 300;
    const centerZ = (Math.random() - 0.5) * 200;
    
    for (let i = 0; i < segment.points; i++) {
      points.push({
        x: centerX + (Math.random() - 0.5) * 100,
        y: centerY + (Math.random() - 0.5) * 100,
        z: centerZ + (Math.random() - 0.5) * 80,
        segment: segment.id,
        color: segment.color,
      });
    }
  });
  
  return points;
};

export function Vector3D() {
  const [zoom, setZoom] = useState(1);
  const [rotation, setRotation] = useState({ x: 15, y: 30 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const pointsRef = useRef(generatePoints());

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Transform and project 3D points to 2D
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const angleX = (rotation.x * Math.PI) / 180;
    const angleY = (rotation.y * Math.PI) / 180;

    const projectedPoints = pointsRef.current.map((point) => {
      // Rotate around Y axis
      let x = point.x * Math.cos(angleY) - point.z * Math.sin(angleY);
      let z = point.x * Math.sin(angleY) + point.z * Math.cos(angleY);
      
      // Rotate around X axis
      const y = point.y * Math.cos(angleX) - z * Math.sin(angleX);
      z = point.y * Math.sin(angleX) + z * Math.cos(angleX);

      // Project to 2D
      const scale = 400 / (400 + z);
      const x2d = centerX + x * scale * zoom;
      const y2d = centerY + y * scale * zoom;
      
      return {
        x: x2d,
        y: y2d,
        z: z,
        color: point.color,
        size: scale * 4,
      };
    });

    // Sort by z-index (far to near)
    projectedPoints.sort((a, b) => a.z - b.z);

    // Draw points
    projectedPoints.forEach((point) => {
      ctx.fillStyle = point.color;
      ctx.globalAlpha = 0.7;
      ctx.beginPath();
      ctx.arc(point.x, point.y, point.size, 0, Math.PI * 2);
      ctx.fill();
    });

    ctx.globalAlpha = 1;
  }, [zoom, rotation]);

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    
    const deltaX = e.clientX - dragStart.x;
    const deltaY = e.clientY - dragStart.y;
    
    setRotation({
      x: rotation.x + deltaY * 0.5,
      y: rotation.y + deltaX * 0.5,
    });
    
    setDragStart({ x: e.clientX, y: e.clientY });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="text-sm text-gray-500">
          Cliquez sur un point pour voir le détail
        </div>
        
        {/* Color selector buttons */}
        <div className="flex gap-1.5">
          {segments.map((segment) => (
            <button
              key={segment.id}
              className="w-8 h-8 rounded-md border-2 border-white shadow-sm hover:scale-110 transition-transform"
              style={{ backgroundColor: segment.color }}
              title={segment.name}
            />
          ))}
        </div>
      </div>

      {/* Canvas */}
      <div className="relative bg-white rounded-lg border border-gray-200 overflow-hidden">
        <canvas
          ref={canvasRef}
          width={800}
          height={500}
          className="w-full cursor-move"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        />
        
        {/* Axis labels */}
        <div className="absolute bottom-4 left-4 text-xs text-gray-400">Y</div>
        <div className="absolute bottom-4 right-4 text-xs text-gray-400">X</div>
        <div className="absolute top-4 left-4 text-xs text-gray-400">Z</div>
      </div>

      {/* Legend and Zoom */}
      <div className="flex items-center justify-between">
        <div className="flex flex-wrap gap-x-4 gap-y-1.5 text-xs">
          {segments.map((segment) => (
            <div key={segment.id} className="flex items-center gap-1.5">
              <div
                className="w-2.5 h-2.5 rounded-full"
                style={{ backgroundColor: segment.color }}
              />
              <span className="text-gray-600">{segment.name}</span>
            </div>
          ))}
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setZoom(Math.max(0.5, zoom - 0.1))}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            title="Zoom out"
          >
            <ZoomOut className="size-4 text-gray-600" />
          </button>
          <span className="text-xs text-gray-500 w-12 text-center">ZOOM</span>
          <button
            onClick={() => setZoom(Math.min(2, zoom + 0.1))}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            title="Zoom in"
          >
            <ZoomIn className="size-4 text-gray-600" />
          </button>
        </div>
      </div>
    </div>
  );
}