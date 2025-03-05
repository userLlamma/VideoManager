import React, { useState } from 'react';
import { Upload, Button, Input, Select, Table, message, Tabs } from 'antd';
import { UploadOutlined, SearchOutlined } from '@ant-design/icons';

const { TabPane } = Tabs;
const { Option } = Select;

const VideoMaterialTestApp = () => {
  // 状态管理
  const [file, setFile] = useState(null);
  const [searchParams, setSearchParams] = useState({
    search: '',
    tagIds: [],
    skip: 0,
    limit: 20
  });
  const [projectName, setProjectName] = useState('');
  const [projectDescription, setProjectDescription] = useState('');
  const [materials, setMaterials] = useState([]);
  const [tags, setTags] = useState([]);
  const [projects, setProjects] = useState([]);

  // 上传视频处理
  const handleUpload = async () => {
    if (!file) {
      message.error('请选择文件');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('extract_only', 'false');

    try {
      const response = await fetch('http://localhost:8000/api/v1/processing/upload', {
        method: 'POST',
        body: formData
      });
      const result = await response.json();
      message.success('视频上传成功');
      console.log(result);
    } catch (error) {
      message.error('上传失败');
      console.error(error);
    }
  };

  // 搜索素材
  const searchMaterials = async () => {
    try {
      const queryParams = new URLSearchParams({
        search: searchParams.search,
        tag_ids: searchParams.tagIds.join(','),
        skip: searchParams.skip,
        limit: searchParams.limit
      }).toString();

      const response = await fetch(`http://localhost:8000/api/v1/materials?${queryParams}`);
      const result = await response.json();
      setMaterials(result);
      message.success('素材搜索成功');
    } catch (error) {
      message.error('搜索失败');
      console.error(error);
    }
  };

  // 创建项目
  const createProject = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/v1/projects', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'accept': 'application/json'
        },
        body: JSON.stringify({
          name: projectName,
          description: projectDescription
        })
      });
      const result = await response.json();
      message.success('项目创建成功');
      setProjects([...projects, result]);
    } catch (error) {
      message.error('项目创建失败');
      console.error(error);
    }
  };

  // 素材表格列配置
  const materialColumns = [
    { title: 'ID', dataIndex: 'id', key: 'id' },
    { title: '名称', dataIndex: 'name', key: 'name' },
    { title: '描述', dataIndex: 'description', key: 'description' },
    { 
      title: '操作', 
      key: 'action', 
      render: (text, record) => (
        <Button type="link" onClick={() => console.log('查看素材详情', record)}>
          查看详情
        </Button>
      )
    }
  ];

  return (
    <div className="p-6 bg-gray-100 min-h-screen">
      <h1 className="text-2xl font-bold mb-6">视频素材管理系统 - 测试页面</h1>
      
      <Tabs defaultActiveKey="1">
        {/* 视频上传 */}
        <TabPane tab="视频上传" key="1">
          <div className="bg-white p-4 rounded shadow">
            <Upload 
              beforeUpload={(file) => {
                setFile(file);
                return false; // 阻止自动上传
              }}
              showUploadList={false}
            >
              <Button icon={<UploadOutlined />}>选择视频文件</Button>
            </Upload>
            {file && <p className="mt-2">已选择: {file.name}</p>}
            <Button 
              type="primary" 
              className="mt-4" 
              onClick={handleUpload}
              disabled={!file}
            >
              开始上传
            </Button>
          </div>
        </TabPane>

        {/* 素材搜索 */}
        <TabPane tab="素材搜索" key="2">
          <div className="bg-white p-4 rounded shadow">
            <div className="flex space-x-4 mb-4">
              <Input 
                placeholder="搜索关键词" 
                value={searchParams.search}
                onChange={(e) => setSearchParams({...searchParams, search: e.target.value})}
              />
              <Select 
                mode="multiple" 
                style={{ width: '200px' }} 
                placeholder="选择标签"
                onChange={(values) => setSearchParams({...searchParams, tagIds: values})}
              >
                {/* 这里应该从后端获取标签列表 */}
                <Option value="1">人物</Option>
                <Option value="2">风景</Option>
                <Option value="3">动作</Option>
              </Select>
              <Button 
                type="primary" 
                icon={<SearchOutlined />} 
                onClick={searchMaterials}
              >
                搜索
              </Button>
            </div>
            <Table 
              columns={materialColumns} 
              dataSource={materials} 
              rowKey="id"
              pagination={{
                total: materials.length,
                pageSize: searchParams.limit,
                onChange: (page) => {
                  setSearchParams({
                    ...searchParams, 
                    skip: (page - 1) * searchParams.limit
                  });
                  searchMaterials();
                }
              }}
            />
          </div>
        </TabPane>

        {/* 项目管理 */}
        <TabPane tab="项目管理" key="3">
          <div className="bg-white p-4 rounded shadow">
            <div className="flex space-x-4 mb-4">
              <Input 
                placeholder="项目名称" 
                value={projectName}
                onChange={(e) => setProjectName(e.target.value)}
              />
              <Input 
                placeholder="项目描述" 
                value={projectDescription}
                onChange={(e) => setProjectDescription(e.target.value)}
              />
              <Button 
                type="primary" 
                onClick={createProject}
                disabled={!projectName}
              >
                创建项目
              </Button>
            </div>
            <Table 
              columns={[
                { title: 'ID', dataIndex: 'id', key: 'id' },
                { title: '名称', dataIndex: 'name', key: 'name' },
                { title: '描述', dataIndex: 'description', key: 'description' }
              ]} 
              dataSource={projects} 
              rowKey="id"
            />
          </div>
        </TabPane>
      </Tabs>
    </div>
  );
};

export default VideoMaterialTestApp;