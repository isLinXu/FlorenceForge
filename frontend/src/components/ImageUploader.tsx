import { useCallback, useRef, useState } from "react";

interface ImageUploaderProps {
  onImageChange: (file: File | null) => void;
}

const ACCEPTED_TYPES = [
  "image/png",
  "image/jpeg",
  "image/webp",
  "image/gif",
];

export function ImageUploader({ onImageChange }: ImageUploaderProps) {
  const [preview, setPreview] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File) => {
      if (!ACCEPTED_TYPES.includes(file.type)) {
        return;
      }

      const reader = new FileReader();
      reader.onloadend = () => {
        const base64 = reader.result as string;
        setPreview(base64);
        onImageChange(file);
      };
      reader.readAsDataURL(file);
    },
    [onImageChange]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);

      const file = e.dataTransfer.files?.[0];
      if (file) {
        handleFile(file);
      }
    },
    [handleFile]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleClick = useCallback(() => {
    inputRef.current?.click();
  }, []);

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        handleFile(file);
      }
    },
    [handleFile]
  );

  const handleRemove = useCallback(() => {
    setPreview(null);
    onImageChange(null);
    if (inputRef.current) {
      inputRef.current.value = "";
    }
  }, [onImageChange]);

  return (
    <div className="w-full">
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_TYPES.join(",")}
        onChange={handleInputChange}
        className="hidden"
      />

      {preview ? (
        <div className="relative group">
          <img
            src={preview}
            alt="Preview"
            className="w-full h-64 object-contain rounded-xl border border-warm-200 bg-warm-50"
          />
          <button
            onClick={handleRemove}
            className="absolute top-3 right-3 p-2 rounded-lg bg-warm-800/80 text-warm-100 opacity-0 group-hover:opacity-100 transition-opacity duration-200 hover:bg-warm-900"
            title="Remove image"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M18 6 6 18" />
              <path d="m6 6 12 12" />
            </svg>
          </button>
          <div className="absolute bottom-3 left-3 px-3 py-1.5 rounded-lg bg-warm-800/80 text-warm-100 text-xs font-medium opacity-0 group-hover:opacity-100 transition-opacity duration-200">
            Click to replace
          </div>
          <div
            onClick={handleClick}
            className="absolute inset-0 cursor-pointer"
          />
        </div>
      ) : (
        <div
          onClick={handleClick}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          className={`
            relative w-full h-64 rounded-xl border-2 border-dashed cursor-pointer
            transition-all duration-200 flex flex-col items-center justify-center gap-4
            ${
              isDragging
                ? "border-primary-400 bg-primary-50/50"
                : "border-warm-300 bg-warm-50 hover:border-warm-400 hover:bg-warm-100"
            }
          `}
        >
          <div
            className={`
              p-4 rounded-full transition-colors duration-200
              ${isDragging ? "bg-primary-100" : "bg-warm-200"}
            `}
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="28"
              height="28"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              className={`
                transition-colors duration-200
                ${isDragging ? "text-primary-600" : "text-warm-500"}
              `}
            >
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" x2="12" y1="3" y2="15" />
            </svg>
          </div>
          <div className="text-center">
            <p className="text-sm font-medium text-warm-700">
              {isDragging ? "Drop image here" : "Drag & drop an image"}
            </p>
            <p className="text-xs text-warm-500 mt-1">
              or click to browse · PNG, JPG, WebP, GIF
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
