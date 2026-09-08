/**
 * AI Photo Selection System — Main Alpine.js application
 */
function photoApp() {
  return {
    // ── State ────────────────────────────────────────────
    page: 'upload',
    systemOnline: true,
    mlMode: 'cold_start',

    // Ingestion & Upload
    ingestMode: 'local', // 'local' | 'upload'
    activeBatch: null,
    newBatchName: '',
    localFolderPath: '',
    recursiveScan: false,
    folderPresets: [],
    batchStatus: 'idle',
    progress: { total: 0, done: 0, errors: 0, failed_technical_filter: 0, pending: 0 },
    uploadQueue: [],
    recentBatches: [],
    progressInterval: null,

    // Review
    reviewBatchId: null,
    reviewData: null,
    reviewLoading: false,
    selections: {},
    viewMode: 'shortlist',
    expandedClusters: {},
    photoMetaMap: {},

    // Dashboard
    dashStats: null,
    modelVersions: [],
    configData: null,

    // Toast
    toast: { show: false, message: '' },

    // ── Init ──────────────────────────────────────────────
    async init() {
      await this.loadRecentBatches();
      await this.loadFolderPresets();
      await this.loadDashboardStats();
      this.checkSystem();
    },

    async checkSystem() {
      try {
        const r = await fetch('/api/stats');
        if (r.ok) {
          const data = await r.json();
          this.systemOnline = true;
          this.mlMode = data.ml_mode || 'cold_start';
        }
      } catch { this.systemOnline = false; }
    },

    async loadFolderPresets() {
      try {
        const r = await fetch('/api/folder-presets');
        if (r.ok) this.folderPresets = await r.json();
      } catch {}
    },

    // ── Computed ──────────────────────────────────────────
    get progressPercent() {
      if (!this.progress.total) return 0;
      return Math.round(((this.progress.done + this.progress.errors) / this.progress.total) * 100);
    },

    // ── Batch management ──────────────────────────────────
    async createBatch() {
      if (!this.newBatchName.trim()) {
        this.showToast('กรุณาระบุชื่อรายงาน');
        return;
      }
      try {
        const r = await fetch(`/api/batches?report_name=${encodeURIComponent(this.newBatchName)}`, { method: 'POST' });
        const data = await r.json();
        this.activeBatch = { id: data.batch_id || data.id, report_name: data.report_name };
        this.batchStatus = 'active';
        this.progress = { total: 0, done: 0, errors: 0, failed_technical_filter: 0, pending: 0 };
        this.uploadQueue = [];
        this.showToast(`✅ สร้าง Batch: ${this.activeBatch.id?.slice(0, 8)}...`);
        await this.loadRecentBatches();

        // Start polling progress
        this.startProgressPolling();
      } catch (e) {
        this.showToast('❌ สร้าง Batch ล้มเหลว: ' + e.message);
      }
    },

    // ── Direct Local Folder Scan ───────────────────────────
    applyPreset(path) {
      this.localFolderPath = path;
    },

    async scanActiveBatchFolder() {
      if (!this.activeBatch?.id) {
        this.showToast('กรุณาสร้าง Batch ก่อน');
        return;
      }
      if (!this.localFolderPath.trim()) {
        this.showToast('กรุณาระบุที่อยู่โฟลเดอร์ในเครื่อง');
        return;
      }

      try {
        const r = await fetch(`/api/batches/${this.activeBatch.id}/scan-folder`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            folder_path: this.localFolderPath.trim(),
            recursive: this.recursiveScan,
            auto_cluster: true,
          }),
        });

        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || 'ไม่สามารถสแกนโฟลเดอร์ได้');

        this.showToast(`🔍 ${data.message}`);
        this.batchStatus = 'scanning';
        this.startProgressPolling();
      } catch (e) {
        this.showToast('❌ สแกนล้มเหลว: ' + e.message);
      }
    },

    async quickScanLocal() {
      if (!this.localFolderPath.trim()) {
        this.showToast('กรุณาระบุที่อยู่โฟลเดอร์ในเครื่อง');
        return;
      }

      const reportName = this.newBatchName.trim() || `สแกนโฟลเดอร์ ${this.localFolderPath.split(/[/\\]/).filter(Boolean).pop() || ''}`;

      try {
        const r = await fetch('/api/scan-local', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            report_name: reportName,
            folder_path: this.localFolderPath.trim(),
            recursive: this.recursiveScan,
          }),
        });

        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || 'ไม่สามารถสแกนโฟลเดอร์ได้');

        this.activeBatch = { id: data.batch_id || data.id, report_name: data.report_name };
        this.batchStatus = 'scanning';
        this.progress = { total: data.photos_found, done: 0, errors: 0, failed_technical_filter: 0, pending: data.photos_found };
        this.showToast(`🚀 ${data.message}`);
        await this.loadRecentBatches();
        this.startProgressPolling();
      } catch (e) {
        this.showToast('❌ สแกนล้มเหลว: ' + e.message);
      }
    },

    startProgressPolling() {
      if (this.progressInterval) clearInterval(this.progressInterval);
      this.progressInterval = setInterval(async () => {
        if (!this.activeBatch?.id) { clearInterval(this.progressInterval); return; }
        await this.refreshProgress();
      }, 1500);
    },

    async refreshProgress() {
      if (!this.activeBatch?.id) return;
      try {
        const r = await fetch(`/api/batches/${this.activeBatch.id}/progress`);
        if (r.ok) {
          this.progress = await r.json();
          // If total > 0 and pending == 0, marked finalized
          if (this.progress.total > 0 && this.progress.pending === 0 && this.batchStatus === 'scanning') {
            this.batchStatus = 'done';
          }
        }
      } catch {}
    },

    async completeBatch() {
      if (!this.activeBatch?.id) return;
      try {
        await fetch(`/api/batches/${this.activeBatch.id}/complete`, { method: 'POST' });
        this.batchStatus = 'finalizing';
        this.showToast('⚙ กำลัง Cluster ภาพ... รอสักครู่');
        await this.loadRecentBatches();
      } catch (e) {
        this.showToast('❌ ' + e.message);
      }
    },

    async deleteBatch(batchId, event = null) {
      if (event) event.stopPropagation();
      if (!confirm(`คุณต้องการลบ Batch นี้ (${batchId?.slice(0, 8)}...) และข้อมูลภาพทั้งหมดใช่หรือไม่?`)) return;

      try {
        const r = await fetch(`/api/batches/${batchId}`, { method: 'DELETE' });
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || 'ไม่สามารถลบ Batch ได้');

        this.showToast(`🗑 ${data.message || 'ลบ Batch สำเร็จ'}`);

        if (this.activeBatch?.id === batchId) {
          this.activeBatch = null;
          this.batchStatus = 'idle';
          if (this.progressInterval) clearInterval(this.progressInterval);
        }
        if (this.reviewBatchId === batchId) {
          this.reviewBatchId = null;
          this.reviewData = null;
        }

        await this.loadRecentBatches();
        await this.loadDashboardStats();
      } catch (e) {
        this.showToast('❌ ลบไม่สำเร็จ: ' + e.message);
      }
    },

    async loadRecentBatches() {
      try {
        const r = await fetch('/api/batches');
        if (r.ok) this.recentBatches = await r.json();
      } catch {}
    },

    // ── File upload ──────────────────────────────────────
    handleDrop(event) {
      const files = [...event.dataTransfer.files];
      this.uploadFiles(files);
    },

    handleFileInput(event) {
      const files = [...event.target.files];
      this.uploadFiles(files);
      // Reset input value so same files can be re-selected if needed
      event.target.value = '';
    },

    async uploadFiles(files) {
      if (!files || !files.length) return;

      // Filter supported image types (case-insensitive)
      const supported = files.filter(f =>
        /\.(jpe?g|png|heic|heif|tiff?|webp|bmp)$/i.test(f.name) ||
        (f.type && f.type.startsWith('image/'))
      );

      if (!supported.length) {
        this.showToast('❌ ไม่พบไฟล์ภาพที่รองรับ (รองรับ .JPG, .PNG, .HEIC)');
        return;
      }

      // Auto-create batch if not exists
      if (!this.activeBatch) {
        const batchName = this.newBatchName.trim() || `อัปโหลด ${new Date().toLocaleTimeString('th-TH')}`;
        try {
          const r = await fetch(`/api/batches?report_name=${encodeURIComponent(batchName)}`, { method: 'POST' });
          const data = await r.json();
          this.activeBatch = { id: data.batch_id || data.id, report_name: data.report_name };
          this.batchStatus = 'active';
          this.progress = { total: 0, done: 0, errors: 0, failed_technical_filter: 0, pending: 0 };
          this.uploadQueue = [];
          this.startProgressPolling();
          await this.loadRecentBatches();
        } catch (e) {
          this.showToast('❌ สร้าง Batch ล้มเหลว: ' + e.message);
          return;
        }
      }

      this.showToast(`⬆ กำลังเริ่มอัปโหลด ${supported.length} ภาพ...`);

      // Chunk into batches of 10
      const CHUNK = 10;
      for (let i = 0; i < supported.length; i += CHUNK) {
        const batch = supported.slice(i, i + CHUNK);
        await this.uploadBatch(batch);
      }
    },

    async uploadBatch(files) {
      const fd = new FormData();
      files.forEach(f => fd.append('files', f));

      files.forEach(f => this.uploadQueue.push({ name: f.name, status: 'uploading' }));

      try {
        const r = await fetch(`/api/upload/${this.activeBatch.id}`, {
          method: 'POST',
          body: fd,
        });
        const data = await r.json();

        data.results?.forEach(res => {
          const item = this.uploadQueue.find(q => q.name === res.filename);
          if (item) item.status = res.status === 'queued' ? 'done' : 'error';
        });

        await this.refreshProgress();
      } catch (e) {
        files.forEach(f => {
          const item = this.uploadQueue.find(q => q.name === f.name);
          if (item) item.status = 'error';
        });
        this.showToast('❌ Upload ล้มเหลว: ' + e.message);
      }
    },

    // ── Review ──────────────────────────────────────────
    async loadBatchForReview(batchId) {
      this.page = 'review';
      this.reviewBatchId = batchId;
      this.reviewLoading = true;
      this.reviewData = null;
      this.selections = {};
      this.expandedClusters = {};
      this.photoMetaMap = {};

      try {
        const [reviewR, selectionsR] = await Promise.all([
          fetch(`/api/batches/${batchId}/review`),
          fetch(`/api/batches/${batchId}/selections`),
        ]);

        if (reviewR.ok) {
          this.reviewData = await reviewR.json();
          // Build photo meta lookup
          this.reviewData.clusters?.forEach(c =>
            c.photos?.forEach(p => { this.photoMetaMap[p.id] = p; })
          );
          // Expand first cluster by default
          if (this.reviewData.clusters?.length) {
            this.expandedClusters[this.reviewData.clusters[0].id] = true;
          }
        }
        if (selectionsR.ok) this.selections = await selectionsR.json();
      } catch (e) {
        this.showToast('❌ โหลดข้อมูลล้มเหลว: ' + e.message);
      } finally {
        this.reviewLoading = false;
      }
    },

    goReview() {
      if (this.activeBatch) this.loadBatchForReview(this.activeBatch.id);
    },

    toggleSelection(photoId) {
      if (this.selections[photoId] === true) {
        this.selections[photoId] = false;
      } else if (this.selections[photoId] === false) {
        delete this.selections[photoId];
      } else {
        this.selections[photoId] = true;
      }
      // Force reactivity
      this.selections = { ...this.selections };
    },

    toggleCluster(clusterId) {
      this.expandedClusters[clusterId] = !this.expandedClusters[clusterId];
      this.expandedClusters = { ...this.expandedClusters };
    },

    getPhotoMeta(pid) {
      return this.photoMetaMap[pid] || null;
    },

    async saveSelections() {
      if (!this.reviewBatchId) return;
      try {
        const r = await fetch(`/api/batches/${this.reviewBatchId}/selections`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ selections: this.selections }),
        });
        if (r.ok) {
          const data = await r.json();
          this.showToast(`✅ บันทึกแล้ว ${data.saved} รายการ`);
        }
      } catch (e) {
        this.showToast('❌ บันทึกล้มเหลว: ' + e.message);
      }
    },

    // ── Dashboard ────────────────────────────────────────
    async loadDashboard() {
      await this.loadDashboardStats();
      await this.loadModelVersions();
      await this.loadConfig();
    },

    async loadDashboardStats() {
      try {
        const r = await fetch('/api/stats');
        if (r.ok) {
          this.dashStats = await r.json();
          this.mlMode = this.dashStats.ml_mode || 'cold_start';
        }
      } catch {}
    },

    async loadModelVersions() {
      try {
        const r = await fetch('/api/model-versions');
        if (r.ok) this.modelVersions = await r.json();
      } catch {}
    },

    async loadConfig() {
      try {
        const r = await fetch('/api/config');
        if (r.ok) this.configData = await r.json();
      } catch {}
    },

    async triggerRetrain() {
      try {
        await fetch('/api/retrain', { method: 'POST' });
        this.showToast('🔁 เริ่ม Retraining ใน background...');
        setTimeout(() => this.loadDashboard(), 5000);
      } catch (e) {
        this.showToast('❌ ' + e.message);
      }
    },

    // ── Toast ────────────────────────────────────────────
    showToast(message, duration = 3000) {
      this.toast = { show: true, message };
      setTimeout(() => { this.toast.show = false; }, duration);
    },
  };
}
