import { SystemStatusCard } from "@/components/documents/SystemStatusCard";
import { UploadDropzone }   from "@/components/documents/UploadDropzone";
import { IngestPanel }      from "@/components/documents/IngestPanel";
import { IngestJobLog }     from "@/components/documents/IngestJobLog";
import { DocumentTable }    from "@/components/documents/DocumentTable";
import { useDocuments }     from "@/hooks/useDocuments";
import { useIngestJob }     from "@/hooks/useIngestJob";
import type { AppSettings } from "@/types/api";

interface Props {
  settings: AppSettings;
}

export function DocumentsPage({ settings }: Props) {
  const {
    documents, loading, error,
    uploads, deleting,
    fetchDocuments, upload, remove, clearUploadState,
  } = useDocuments(settings);

  const {
    activeJob, jobHistory, isRunning, error: ingestError,
    triggerAll, fetchJobHistory,
  } = useIngestJob(settings);

  const isUploading  = uploads.some((u) => u.status === "uploading");
  const pendingCount = documents.filter((d) => d.status === "uploaded").length;

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-3xl px-4 py-6 space-y-5">

        <SystemStatusCard settings={settings} />

        <UploadDropzone
          uploads={uploads}
          isUploading={isUploading}
          onUpload={upload}
          onClear={clearUploadState}
        />

        <IngestPanel
          isRunning={isRunning}
          activeJob={activeJob}
          error={ingestError}
          onTrigger={triggerAll}
          pendingCount={pendingCount}
        />

        <DocumentTable
          documents={documents}
          loading={loading}
          error={error}
          deleting={deleting}
          onDelete={remove}
          onRefresh={fetchDocuments}
        />

        <IngestJobLog
          jobs={jobHistory}
          onRefresh={fetchJobHistory}
        />

      </div>
    </div>
  );
}
