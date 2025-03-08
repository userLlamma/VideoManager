<template>
  <div class="app">
    <header class="app-header">
      <div class="header-content">
        <router-link to="/" class="app-logo">
          视频素材管理系统
        </router-link>
        <nav class="app-nav">
          <router-link to="/" class="nav-link">首页</router-link>
          <router-link to="/materials" class="nav-link">素材库</router-link>
          <router-link to="/upload" class="nav-link">上传</router-link>
          <router-link to="/projects" class="nav-link">项目</router-link>
        </nav>
      </div>
    </header>
    
    <main class="app-content">
      <router-view />
    </main>
    
    <footer class="app-footer">
      <div class="footer-content">
        <p>&copy; {{ new Date().getFullYear() }} 视频素材管理系统
          const tagFilter = ref('');
      const projectFilter = ref('');

      // Computed properties
      const descriptionChanged = computed(() => {
        return description.value !== props.material.description;
      });

      const filteredTags = computed(() => {
        const currentTagIds = materialData.value.tags.map(t => t.id);
        const filter = tagFilter.value.toLowerCase();
        
        return tagStore.tags.filter(tag => 
          !currentTagIds.includes(tag.id) && 
          (filter === '' || tag.name.toLowerCase().includes(filter))
        );
      });

      const filteredProjects = computed(() => {
        const currentProjectIds = materialData.value.projects || [];
        const filter = projectFilter.value.toLowerCase();
        
        return projectStore.projects.filter(project => 
          !currentProjectIds.includes(project.id) && 
          (filter === '' || project.name.toLowerCase().includes(filter))
        );
      });

      // Methods
      function close() {
        emit('close');
      }

      async function loadMaterial() {
        loading.value = true;
        error.value = null;
        
        try {
          const material = await materialStore.getMaterial(props.material.id);
          if (material) {
            materialData.value = material;
            description.value = material.description || '';
          } else {
            error.value = 'Failed to load material details';
          }
        } catch (e) {
          error.value = 'An error occurred while loading material';
          console.error(e);
        } finally {
          loading.value = false;
        }
      }

      function getSourceName(path: string): string {
        if (!path) return 'Unknown';
        const parts = path.split('/');
        return parts[parts.length - 1];
      }

      function formatTimestamp(timestamp: number): string {
        const hours = Math.floor(timestamp / 3600);
        const minutes = Math.floor((timestamp % 3600) / 60);
        const seconds = Math.floor(timestamp % 60);
        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
      }

      function formatDate(dateString: string): string {
        if (!dateString) return '';
        const date = new Date(dateString);
        return new Intl.DateTimeFormat('zh-CN', {
          year: 'numeric',
          month: '2-digit',
          day: '2-digit',
          hour: '2-digit',
          minute: '2-digit'
        }).format(date);
      }

      function getProjectName(projectId: number): string {
        const project = projectStore.projects.find(p => p.id === projectId);
        return project ? project.name : `Project #${projectId}`;
      }

      async function saveDescription() {
        if (!descriptionChanged.value || saving.value) return;
        
        saving.value = true;
        
        try {
          const updated = await materialStore.updateMaterial(
            materialData.value.id, 
            { description: description.value }
          );
          
          if (updated) {
            materialData.value = { 
              ...materialData.value, 
              description: description.value 
            };
            emit('updated', materialData.value);
          }
        } catch (e) {
          console.error('Failed to update description:', e);
        } finally {
          saving.value = false;
        }
      }

      async function removeTag(tagId: number) {
        try {
          const success = await materialsApi.removeTag(materialData.value.id, tagId);
          if (success) {
            materialData.value = {
              ...materialData.value,
              tags: materialData.value.tags.filter(t => t.id !== tagId)
            };
            emit('updated', materialData.value);
          }
        } catch (e) {
          console.error('Failed to remove tag:', e);
        }
      }

      async function addTag(tagId: number) {
        try {
          const success = await materialsApi.addTag(materialData.value.id, [tagId]);
          if (success) {
            // Reload material to get updated tags
            const material = await materialStore.getMaterial(materialData.value.id);
            if (material) {
              materialData.value = material;
              emit('updated', material);
            }
          }
          showTagSelector.value = false;
        } catch (e) {
          console.error('Failed to add tag:', e);
        }
      }

      async function createTag(name: string) {
        try {
          const tag = await tagStore.createTag(name);
          if (tag) {
            addTag(tag.id);
          }
        } catch (e) {
          console.error('Failed to create tag:', e);
        }
      }

      async function addToProject(projectId: number) {
        try {
          const success = await projectsApi.addMaterialsToProject(
            projectId, 
            [materialData.value.id]
          );
          
          if (success) {
            // Reload material to get updated projects
            const material = await materialStore.getMaterial(materialData.value.id);
            if (material) {
              materialData.value = material;
              emit('updated', material);
            }
          }
          showProjectSelector.value = false;
        } catch (e) {
          console.error('Failed to add to project:', e);
        }
      }

      async function createProject(name: string) {
        try {
          const project = await projectStore.createProject(name);
          if (project) {
            addToProject(project.id);
          }
        } catch (e) {
          console.error('Failed to create project:', e);
        }
      }

      function confirmDelete() {
        showDeleteConfirm.value = true;
      }

      async function deleteMaterial() {
        if (deleting.value) return;
        
        deleting.value = true;
        
        try {
          const success = await materialStore.deleteMaterial(materialData.value.id);
          if (success) {
            emit('deleted', materialData.value.id);
            showDeleteConfirm.value = false;
          }
        } catch (e) {
          console.error('Failed to delete material:', e);
        } finally {
          deleting.value = false;
        }
      }

      // Watchers
      watch(() => props.material, (newMaterial) => {
        if (newMaterial && newMaterial.id) {
          materialData.value = { ...newMaterial };
          description.value = newMaterial.description || '';
        }
      });

      // Load stores data on mount
      onMounted(() => {
        if (tagStore.tags.length === 0) {
          tagStore.fetchTags();
        }
        
        if (projectStore.projects.length === 0) {
          projectStore.fetchProjects();
        }
      });
      </script>

<style scoped>
.material-detail-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.material-detail-modal {
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
  width: 90%;
  max-width: 800px;
  max-height: 90vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  background-color: #f5f5f5;
  border-bottom: 1px solid #e0e0e0;
}

.modal-header h2 {
  margin: 0;
  font-size: 1.4em;
}

.close-button {
  background: none;
  border: none;
  font-size: 24px;
  cursor: pointer;
  line-height: 1;
  padding: 0;
  color: #666;
}

.close-button:hover {
  color: #000;
}

.modal-content {
  padding: 20px;
  overflow-y: auto;
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.material-preview {
  text-align: center;
  max-height: 300px;
  overflow: hidden;
  border-radius: 4px;
  background-color: #f0f0f0;
}

.material-preview img {
  max-width: 100%;
  max-height: 300px;
  object-fit: contain;
}

.material-info {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.info-row {
  display: flex;
  gap: 10px;
}

.info-label {
  font-weight: bold;
  min-width: 80px;
}

.description-container {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.description-input {
  min-height: 80px;
  padding: 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
  resize: vertical;
}

.save-button {
  align-self: flex-end;
  background-color: #2196f3;
  color: white;
  border: none;
  border-radius: 4px;
  padding: 6px 12px;
  cursor: pointer;
}

.save-button:disabled {
  background-color: #bdbdbd;
  cursor: not-allowed;
}

.tags-container, .projects-container {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.tags-list, .projects-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.tag, .project {
  display: flex;
  align-items: center;
  background-color: #e1f5fe;
  border-radius: 16px;
  padding: 4px 10px;
}

.remove-tag {
  background: none;
  border: none;
  color: #666;
  cursor: pointer;
  font-size: 16px;
  line-height: 1;
  padding: 0 2px;
  margin-left: 4px;
}

.remove-tag:hover {
  color: #f44336;
}

.add-tag-button, .add-project-button {
  background-color: #f5f5f5;
  border: 1px solid #ddd;
  border-radius: 50%;
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  font-weight: bold;
}

.modal-footer {
  padding: 16px 20px;
  display: flex;
  justify-content: flex-end;
  border-top: 1px solid #e0e0e0;
}

.delete-button {
  background-color: #f44336;
  color: white;
  border: none;
  border-radius: 4px;
  padding: 8px 16px;
  cursor: pointer;
}

.delete-button:hover {
  background-color: #d32f2f;
}

.loading-container, .error-container {
  padding: 40px;
  text-align: center;
}

.loader {
  border: 3px solid #f3f3f3;
  border-top: 3px solid #3498db;
  border-radius: 50%;
  width: 30px;
  height: 30px;
  animation: spin 1s linear infinite;
  margin: 0 auto 20px;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.retry-button {
  background-color: #2196f3;
  color: white;
  border: none;
  border-radius: 4px;
  padding: 8px 16px;
  cursor: pointer;
  margin-top: 10px;
}

/* Selector overlay styles */
.selector-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1100;
}

.selector-modal {
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
  width: 90%;
  max-width: 400px;
  max-height: 90vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.selector-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background-color: #f5f5f5;
  border-bottom: 1px solid #e0e0e0;
}

.selector-header h3 {
  margin: 0;
  font-size: 1.2em;
}

.selector-content {
  padding: 16px;
  overflow-y: auto;
  max-height: 400px;
}

.selector-input {
  width: 100%;
  padding: 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
  margin-bottom: 12px;
}

.selector-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.selector-item {
  padding: 8px 12px;
  border-radius: 4px;
  background-color: #f5f5f5;
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.selector-item:hover {
  background-color: #e1f5fe;
}

.item-category, .item-count {
  font-size: 0.8em;
  color: #666;
  background-color: rgba(0, 0, 0, 0.05);
  padding: 2px 6px;
  border-radius: 10px;
}

.no-items {
  text-align: center;
  padding: 20px;
  color: #666;
}

.create-button {
  background-color: #2196f3;
  color: white;
  border: none;
  border-radius: 4px;
  padding: 6px 12px;
  cursor: pointer;
  margin-top: 10px;
}

.confirm-modal {
  max-width: 350px;
}

.confirm-buttons {
  display: flex;
  gap: 12px;
  justify-content: flex-end;
  margin-top: 20px;
}

.cancel-button {
  background-color: #e0e0e0;
  border: none;
  border-radius: 4px;
  padding: 8px 16px;
  cursor: pointer;
}

.confirm-delete-button {
  background-color: #f44336;
  color: white;
  border: none;
  border-radius: 4px;
  padding: 8px 16px;
  cursor: pointer;
}

.confirm-delete-button:disabled {
  background-color: #ef9a9a;
  cursor: not-allowed;
}
</style>